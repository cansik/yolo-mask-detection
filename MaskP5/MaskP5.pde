import ch.bildspur.vision.*;
import ch.bildspur.vision.result.*;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;
import processing.video.Capture;

import java.nio.file.Paths;
import java.nio.file.Path;
import java.util.List;

Capture cam;
PImage inputImage;

DeepVision deepVision = new DeepVision(this);
YOLONetwork yolo;
List<ObjectDetectionResult> detections;

// speed of the yolo algorithm
int detectionSize = 128;

public void setup() {
  size(640, 480, FX2D);
  frameRate(30);
  colorMode(HSB, 360, 100, 100);

  println("creating model...");
  Path model = Paths.get(sketchPath("../models/mask-yolov3-tiny-prn.cfg")).toAbsolutePath();
  Path weights = Paths.get(sketchPath("../models/mask-yolov3-tiny-prn.weights")).toAbsolutePath();

  yolo = new YOLONetwork(model, weights, detectionSize, detectionSize);
  yolo.setLabels("good", "bad", "none");

  println("loading yolo model...");
  yolo.setup();

  String[] cams = Capture.list();
  println("Cameras: ");
  printArray(cams);

  cam = new Capture(this, 640, 480, cams[0]);
  cam.start();

  inputImage = new PImage(320, 240, RGB);
}

public void draw() {
  background(55);

  if (cam.available()) {
    cam.read();
  }

  inputImage.copy(cam, 0, 0, cam.width, cam.height, 0, 0, inputImage.width, inputImage.height);

  yolo.setConfidenceThreshold(0.50f);
  detections = yolo.run(inputImage);
  cam.filter(GRAY);
  image(cam, 0, 0);

  scale(2);
  for (ObjectDetectionResult detection : detections) {    
    noFill();
    strokeWeight(2f);

    switch(detection.getClassId()) {
    case 0:
      stroke(120, 80, 100);
      break;

    case 1:
      stroke(40, 80, 100);
      break;

    case 2:
      stroke(0, 80, 100);
      break;
    }

    rect(detection.getX(), detection.getY(), detection.getWidth(), detection.getHeight());

    fill(0);
    String label = detection.getClassName();
    text(label + " " + nf(detection.getConfidence(), 0, 2), detection.getX(), detection.getY());
  }

  surface.setTitle("Mask YOLO Test - FPS: " + Math.round(frameRate));
}
