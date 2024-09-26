<template>
    <div class="whiteboard">
      <canvas
        ref="canvas"
        @mousedown="startDrawing"
        @mousemove="draw"
        @mouseup="stopDrawing"
        @mouseleave="stopDrawing"
      ></canvas>
  
      <div class="suggestions" v-if="suggestions.length">
        <h3>Suggestions:</h3>
        <ul>
          <li
            v-for="(suggestion, index) in suggestions"
            :key="index"
            @click="acceptSuggestion(suggestion)"
          >
            {{ suggestion }}
          </li>
        </ul>
      </div>
    </div>
  </template>
  
  <script>
  import * as tensorflow from '@tensorflow/tensorflowjs';
  
  export default {
    data() {
      return {
        context: null,
        drawing: false,
        model: null,
        modelLoaded: false,
        suggestions: [],
      };
    },
    async mounted() {
      try {
        await tensorflow.setBackend('webgl');
        await tensorflow.ready();
        this.model = await tensorflow.loadGraphModel(
          'https://storage.googleapis.com/tensorflowjs-models/savedmodel/ssdlite_mobilenet_v2/model.json'
        );
        this.modelLoaded = true;
      } catch (error) {
        await tensorflow.setBackend('cpu');
        await tensorflow.ready();
        this.model = await tensorflow.loadGraphModel(
          'https://storage.googleapis.com/tensorflowjs-models/savedmodel/ssdlite_mobilenet_v2/model.json'
        );
        this.modelLoaded = true;
      }
  
      const canvas = this.$refs.canvas;
      canvas.width = window.innerWidth * 0.8;
      canvas.height = window.innerHeight * 0.8;
      this.context = canvas.getContext('2d');
      this.context.lineWidth = 2;
    },
  
    methods: {
      startDrawing(event) {
        if (!this.modelLoaded) {
          return;
        }
  
        this.drawing = true;
        this.context.beginPath();
        this.context.moveTo(event.offsetX, event.offsetY);
      },
      draw(event) {
        if (!this.drawing) return;
        this.context.lineTo(event.offsetX, event.offsetY);
        this.context.stroke();
      },
      stopDrawing() {
        if (!this.drawing) return;
        this.drawing = false;
        this.context.closePath();
  
        if (this.modelLoaded) {
          this.predictDrawing();
        }
      },
  
      async predictDrawing() {
        if (!this.model) {
          return;
        }
  
        const imageData = this.context.getImageData(
          0,
          0,
          this.$refs.canvas.width,
          this.$refs.canvas.height
        );
  
        const tensor = tensorflow.browser
          .fromPixels(imageData)
          .resizeNearestNeighbor([224, 224])
          .expandDims(0)
          .toInt();
  
        const normalizedTensor = tensor.div(255.0);
  
        try {
          await normalizedTensor.data();
          const prediction = await this.model.executeAsync({
            image_tensor: normalizedTensor,
          });
  
  
          this.displayPrediction(prediction);
        } catch (err) {
          console.error('Error during prediction:', err);
        }
      },
  
      displayPrediction(prediction) {
        // finish later
        console.log('Prediction:', prediction);
      },
    },
  };
  </script>
  
  <style>
  .whiteboard canvas {
    border: 2px solid #000;
    cursor: crosshair;
  }
  </style>
  