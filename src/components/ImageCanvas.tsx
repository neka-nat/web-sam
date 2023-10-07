import React, { useCallback, useRef, useEffect } from 'react';
import * as ort from 'onnxruntime-web';

type ImageCanvasProps = {
  imageEmbeddings: any;
  imageImageData: ImageData | undefined;
  onStatusChange: (message: string) => void;
}

const ImageCanvas: React.FC<ImageCanvasProps> = ({ imageEmbeddings, imageImageData, onStatusChange }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const handleClick = useCallback(async (event: MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas || !imageImageData) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    console.log('Clicked position:', x, y);
    onStatusChange(`Clicked on (${x}, ${y}). Downloading the decoder model if needed and generating mask...`);

    let context = canvas.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageImageData.width;
    canvas.height = imageImageData.height;
    context.putImageData(imageImageData, 0, 0);
    context.fillStyle = 'green';
    context.fillRect(x, y, 5, 5);
    const pointCoords = new ort.Tensor(new Float32Array([x, y, 0, 0]), [1, 2, 2]);
    const pointLabels = new ort.Tensor(new Float32Array([0, -1]), [1, 2]);
    const maskInput = new ort.Tensor(new Float32Array(256 * 256), [1, 1, 256, 256]);
    const hasMask = new ort.Tensor(new Float32Array([0]), [1,]);
    const originalImageSize = new ort.Tensor(new Float32Array([684, 1024]), [2,]);

    ort.env.wasm.numThreads = 1;
    const decodingSession = await ort.InferenceSession.create('models/mobilesam.decoder.quant.onnx');
    console.log("Decoder session", decodingSession);
    const decodingFeeds = {
        "image_embeddings": imageEmbeddings,
        "point_coords": pointCoords,
        "point_labels": pointLabels,
        "mask_input": maskInput,
        "has_mask_input": hasMask,
        "orig_im_size": originalImageSize
    }

    const start = Date.now();
    try {
        const results = await decodingSession.run(decodingFeeds);
        const mask = results.masks;
        const maskImageData = mask.toImageData();
        context.globalAlpha = 0.5;
        // convert image data to image bitmap
        let imageBitmap = await createImageBitmap(maskImageData);
        context.drawImage(imageBitmap, 0, 0);

    } catch (error) {
        console.log(`caught error: ${error}`)
    }
    const end = Date.now();
    console.log(`generating masks took ${(end - start) / 1000} seconds`);
    onStatusChange(`Mask generated. Click on the image to generate a new mask.`);
  }, [imageEmbeddings, imageImageData, onStatusChange]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageImageData) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    context.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageImageData.width;
    canvas.height = imageImageData.height;
    context.putImageData(imageImageData, 0, 0);
  }, [imageImageData]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener('click', handleClick);

    return () => {
      canvas.removeEventListener('click', handleClick);
    };
  }, [handleClick]);

  return <canvas ref={canvasRef} />;
};

export default ImageCanvas;