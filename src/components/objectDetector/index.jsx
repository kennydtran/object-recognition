import React, { useRef, useState } from "react";
import styled from "styled-components";

import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

const ObjectDetectorContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const DetectorContainer = styled.div`
  min-width: 400px;
  height: 700px;
  background-color: rgb(240, 235, 226);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
  transition: all 260ms ease-in-out;
 
  &:hover {
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
  transform: scale(1.02);
  }
`;

const TargetImg = styled.img`
  height: 100%;
`;

const HiddenFileInput = styled.input`
  display: none;
`;

const SelectButton = styled.button`
  padding: 7px 10px;
  border: 2px solid transparent;
  background-color: rgb(112, 168, 101);
  color: rgb(238, 232, 224);
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
  font-size: 16px;
  border-radius: 20px;
  font-weight: 400;
  outline: none;
  margin-top: 2em;
  cursor: pointer;
  transition: all 260ms ease-in-out;

  &:hover {
    background-color: rgb(132, 192, 120);
  }
`;



const TargetBox = styled.div`
  position: absolute;

  left: ${({ x }) => x + "px"};
  top: ${({ y }) => y + "px"};
  width: ${({ width }) => width + "px"};
  height: ${({ height }) => height + "px"};

  border: 4px solid rgb(0, 255, 0);
  background-color: transparent;
  z-index: 20;

  &::before {
    content: "${({ classType, score }) => `${classType} ${score.toFixed(1)}%`}";
    color:rgb(0, 255, 0);
    font-weight: 400;
    font-size: 17px;
    position: absolute;
    top: -1.5em;
    left: -5px;
  }
`;

export function ObjectDetector(props) {
  const fileInputRef = useRef();
  const imageRef = useRef();
  const [imgData, setImgData] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [isLoading, setLoading] = useState(false);

  const isEmptyPredictions = !predictions || predictions.length === 0;

  const openFilePicker = () => {
    if (fileInputRef.current) fileInputRef.current.click();
  };

  const normalizePredictions = (predictions, imgSize) => {
    if (!predictions || !imgSize || !imageRef) return predictions || [];
    return predictions.map((prediction) => {
      const { bbox } = prediction;
      const oldX = bbox[0];
      const oldY = bbox[1];
      const oldWidth = bbox[2];
      const oldHeight = bbox[3];

      const imgWidth = imageRef.current.width;
      const imgHeight = imageRef.current.height;

      const x = (oldX * imgWidth) / imgSize.width;
      const y = (oldY * imgHeight) / imgSize.height;
      const width = (oldWidth * imgWidth) / imgSize.width;
      const height = (oldHeight * imgHeight) / imgSize.height;

      return { ...prediction, bbox: [x, y, width, height] };
    });
  };

  const detectObjectsOnImage = async (imageElement, imgSize) => {
    const model = await cocoSsd.load({});
    const predictions = await model.detect(imageElement, 6);
    const normalizedPredictions = normalizePredictions(predictions, imgSize);
    setPredictions(normalizedPredictions);
    console.log("Predictions: ", predictions);
  };

  const readImage = (file) => {
    return new Promise((rs, rj) => {
      const fileReader = new FileReader();
      fileReader.onload = () => rs(fileReader.result);
      fileReader.onerror = () => rj(fileReader.error);
      fileReader.readAsDataURL(file);
    });
  };

  const onSelectImage = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
  
    setPredictions([]);
    setLoading(true);
  
    const imgData = await readImage(file);
    setImgData(imgData);
  
    const imageElement = document.createElement("img");
    imageElement.src = imgData;
  
    imageElement.onload = async () => {
      const imgSize = {
        width: imageElement.width,
        height: imageElement.height,
      };
      await detectObjectsOnImage(imageElement, imgSize);
      setLoading(false);
    };
  };
  

  return (
    <ObjectDetectorContainer>
      <DetectorContainer>
        {imgData && <TargetImg src={imgData} ref={imageRef} />}
        {!isEmptyPredictions ? (
        predictions.map((prediction, idx) => (
          <TargetBox
            key={idx}
            x={prediction.bbox[0]}
            y={prediction.bbox[1]}
            width={prediction.bbox[2]}
            height={prediction.bbox[3]}
            classType={prediction.class}
            score={prediction.score * 100}
          />
        ))
      ) : (
        imgData &&
        !isLoading && (
          <div
            style={{
              position: "absolute",
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              padding: "1em 1.5em",
              borderRadius: "12px",
              border: "2px solid rgba(255, 74, 74, 0.95)",
              boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.15)",
              color: "rgb(114, 14, 14)",
              fontWeight: "400",
              fontSize: "18px",
              textAlign: "center",
            }}
          >
            Unable to Identify
          </div>
        )
      )}

      </DetectorContainer>
      <HiddenFileInput
        type="file"
        ref={fileInputRef}
        onChange={onSelectImage}
      />
      <SelectButton onClick={openFilePicker}>
        {isLoading ? "Recognizing..." : "Select Image"}
      </SelectButton>
    </ObjectDetectorContainer>
  );
}
