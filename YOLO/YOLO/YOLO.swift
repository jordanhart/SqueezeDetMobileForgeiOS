import MetalPerformanceShaders
import Forge



let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52] // TODO change to squeezeDet anchors.

/*
  The tiny-yolo-voc network from YOLOv2. https://pjreddie.com/darknet/yolo/

  This implementation is cobbled together from the following sources:

  - https://github.com/pjreddie/darknet
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java
  - https://github.com/allanzelener/YAD2K
*/
class YOLO: NeuralNetwork {
  typealias PredictionType = YOLO.Prediction

  struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

  let model: Model

  public init(device: MTLDevice, inflightBuffers: Int) {
    // Note: YOLO expects the input pixels to be in the range 0-1. Our input
    // texture most likely has pixels with values 0-255. However, since Forge
    // uses .float16 as the channel format the Resize layer will automatically
    // convert the pixels to be between 0 and 1.

    let leaky = MPSCNNNeuronReLU(device: device, a: 0.1)

    let input = Input()

    let phase1 = input
             --> Resize(width: 960, height: 288)
             --> Convolution(kernel: (3, 3), channels: 64, stride:(2,2), activation: leaky, name: "conv1")
             --> MaxPooling(kernel: (2, 2), stride: (2, 2), name: "pool1")
    let fire2Squeeze = phase1 --> Convolution(kernel: (1, 1), channels: 16, stride: (1,1), activation: leaky, name: "fire2_squeeze1x1")
    let fire2Result = Concatenate([
        fire2Squeeze --> Convolution(kernel: (1, 1), channels: 64, stride: (1,1), activation: leaky, name: "fire2_expand1x1"),
        fire2Squeeze --> Convolution(kernel: (3, 3), channels: 64, stride: (1,1), activation: leaky, name: "fire2_expand3x3")
        ])
    let fire3Squeeze = fire2Result --> Convolution(kernel: (1, 1), channels: 16, stride: (1,1), activation: leaky, name: "fire3_squeeze1x1")
    let fire3Result = Concatenate([
        fire3Squeeze --> Convolution(kernel: (1, 1), channels: 64, stride: (1,1), activation: leaky, name: "fire3_expand1x1"),
        fire3Squeeze --> Convolution(kernel: (3, 3), channels: 64, stride: (1,1), activation: leaky, name: "fire3_expand3x3")
        ])
    let pool3 = fire3Result --> MaxPooling(kernel: (2, 2), stride: (2, 2), name: "pool3")
    let fire4Squeeze = pool3 --> Convolution(kernel: (1, 1), channels: 32, stride: (1,1), activation: leaky, name: "fire4_squeeze1x1")
    let fire4Result = Concatenate([
        fire4Squeeze --> Convolution(kernel: (1, 1), channels: 128, stride: (1,1), activation: leaky, name: "fire4_expand1x1"),
        fire4Squeeze --> Convolution(kernel: (3, 3), channels: 128, stride: (1,1), activation: leaky, name: "fire4_expand3x3")
        ])
    let fire5Squeeze = fire4Result --> Convolution(kernel: (1, 1), channels: 32, stride: (1,1), activation: leaky, name: "fire5_squeeze1x1")
    let fire5Result = Concatenate([
        fire5Squeeze --> Convolution(kernel: (1, 1), channels: 128, stride: (1,1), activation: leaky, name: "fire5_expand1x1"),
        fire5Squeeze --> Convolution(kernel: (3, 3), channels: 128, stride: (1,1), activation: leaky, name: "fire5_expand3x3")
        ])

    let pool5 = fire5Result --> MaxPooling(kernel: (2, 2), stride: (2, 2), name: "pool5")
    let fire6Squeeze = pool5  --> Convolution(kernel: (1, 1), channels: 48, stride: (1,1), activation: leaky, name: "fire6_squeeze1x1")
    let fire6Result = Concatenate([
        fire6Squeeze --> Convolution(kernel: (1, 1), channels: 192, stride: (1,1), activation: leaky, name: "fire6_expand1x1"),
        fire6Squeeze --> Convolution(kernel: (3, 3), channels: 192, stride: (1,1), activation: leaky, name: "fire6_expand3x3")
            ])
    let fire7Squeeze = fire6Result  --> Convolution(kernel: (1, 1), channels: 48, stride: (1,1), activation: leaky, name: "fire7_squeeze1x1")
    let fire7Result = Concatenate([
        fire7Squeeze --> Convolution(kernel: (1, 1), channels: 192, stride: (1,1), activation: leaky, name: "fire7_expand1x1"),
        fire7Squeeze --> Convolution(kernel: (3, 3), channels: 192, stride: (1,1), activation: leaky, name: "fire7_expand3x3")
        ])
     let fire8Squeeze = fire7Result  --> Convolution(kernel: (1, 1), channels: 64, stride: (1,1), activation: leaky, name: "fire8_squeeze1x1")
     let fire8Result = Concatenate([
           fire8Squeeze  --> Convolution(kernel: (1, 1), channels: 256, stride: (1,1), activation: leaky, name: "fire8_expand1x1"),
           fire8Squeeze  --> Convolution(kernel: (3, 3), channels: 256, stride: (1,1), activation: leaky, name: "fire8_expand3x3")
            ])
    let fire9Squeeze = fire8Result  --> Convolution(kernel: (1, 1), channels: 64, stride: (1,1), activation: leaky, name: "fire9_squeeze1x1")
    let fire9Result = Concatenate([
        fire9Squeeze  --> Convolution(kernel: (1, 1), channels: 256, stride: (1,1), activation: leaky, name: "fire9_expand1x1"),
        fire9Squeeze  --> Convolution(kernel: (3, 3), channels: 256, stride: (1,1), activation: leaky, name: "fire9_expand3x3")
        ])

    let fire10Squeeze = fire9Result --> Convolution(kernel: (1, 1), channels: 96, stride: (1,1), activation: leaky, name: "fire10_squeeze1x1")
    let fire10Result = Concatenate([
        fire10Squeeze --> Convolution(kernel: (1, 1), channels: 384, stride: (1,1), activation: leaky, name: "fire10_expand1x1"),
        fire10Squeeze --> Convolution(kernel: (3, 3), channels: 384, stride: (1,1), activation: leaky, name: "fire10_expand3x3")
            ])
    let fire11Squeeze = fire10Result --> Convolution(kernel: (1, 1), channels: 96, stride: (1,1), activation: leaky, name: "fire11_squeeze1x1")
    let fire11Result = Concatenate([
            fire11Squeeze --> Convolution(kernel: (1, 1), channels: 384, stride: (1,1), activation: leaky, name: "fire11_expand1x1"),
            fire11Squeeze --> Convolution(kernel: (3, 3), channels: 384, stride: (1,1), activation: leaky, name: "fire11_expand3x3")
    ])
    let output = fire11Result --> Convolution(kernel: (1, 1), channels: 72, stride: (1,1), activation: nil, name: "conv12") //problem is that output is different from yolo, so have to reimpliment prediction function. Impliment _add_interpretation_graph in nn_skeleton.py
    
    
    
    model = Model(input: input, output: output)


    let success = model.compile(device: device, inflightBuffers: inflightBuffers) {
      name, count, type in ParameterLoaderBundle(name: name,
                                                 count: count,
                                                 suffix: type == .weights ? "_W" : "_b",
                                                 ext: "bin")
    }

    if success {
      print(model.summary())
    }
  }

  public func encode(commandBuffer: MTLCommandBuffer, texture sourceTexture: MTLTexture, inflightIndex: Int) {
    model.encode(commandBuffer: commandBuffer, texture: sourceTexture, inflightIndex: inflightIndex)
  }

  public func fetchResult(inflightIndex: Int) -> NeuralNetworkResult<Prediction> {
    let featuresImage = model.outputImage(inflightIndex: inflightIndex)
    let features = featuresImage.toFloatArray()
    assert(features.count == 13*13*128)


    // We only run the convolutional part of YOLO on the GPU. The last part of
    // the process is done on the CPU. It should be possible to do this on the
    // GPU too, but it might not be worth the effort.

    var predictions = [Prediction]()

    let blockSize: Float = 32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 5
    let numClasses = 3
    //my code_________________
    let ANCHOR_PER_GRID = 9
    let CLASSES = numClasses
    let num_class_probs = ANCHOR_PER_GRID * CLASSES
    
    let height = featuresImage.height //18
    
    let width = featuresImage.width //60
    
    let channels = featuresImage.featureChannels //72
    
    
    //features is output image. https://github.com/hollance/Forge/blob/master/Docs/Importing.markdown
    //72 channels, so 18 sets of 4 channels for each image
    
    
    
    // This helper function finds the offset in the features array for a given
    // channel for a particular pixel. (See the comment below.)
    func offsetSqueezeDet(_ channel: Int, _ x: Int, _ y: Int) -> Int {
        let slice = channel / 4
        let indexInSlice = channel - slice*4
        let offset = slice*height*width*4 + y*width*4 + x*4 + indexInSlice
        return offset
    }

    
// convert mpsimage to 3D matrix, height, width, channel.
// either use accelerate for transpose or https://github.com/mattt/Surge , before converting to new matrix
    
    let innerMost : [Float] = Array<Float>(repeatElement(0.0, count: channels))
    let middle : [[Float]] = Array(repeating: innerMost, count: height)
    var tensorflowImage : [[[Float]]] = Array(repeating: middle, count: width)
    
    
    //couldnt get matrix class working with subscript ranges nor can slice ranges with arrays of arrays so putting channel first, since only slice via channel
    for x in 0..<width {
        for y in 0..<height{
            for c in 0..<channels{
                tensorflowImage[x][y][c] = features[offsetSqueezeDet(c, x, y)]
            }
        }
    }
//
//    
//    //convdet probability
//    var inputProbs = Array(tensorflowImage[0..<num_class_probs])
//    
////    let outputProbs = Tensor(input: featuresImage, layer: <#Layer#>)
////        --> Resize(-1, CLASSES)
////        --> Softmax()
////        --> Resize(1, anchors, CLASSES)
//
//    Math.softmax(inputProbs)
//    
//    //convdet confidence
    let num_confidence_scores = ANCHOR_PER_GRID+num_class_probs
//
//    var inputConfidence = tensorflowImage[num_class_probs..<num_confidence_scores]
//    
//    
//    
//    //bounding boxes delta
//    
//    var inputBoundingBoxes = [num_confidence_scores..<tensorflowImage.count]
//    
    let innerMostProbs : [Float] = Array<Float>(repeatElement(0.0, count: num_class_probs))
    let middleProbs : [[Float]] = Array(repeating: innerMostProbs, count: height)
    var pred_class_probs : [[[Float]]] = Array(repeating: middleProbs, count: width)
    
    let innerMostConf : [Float] = Array<Float>(repeatElement(0.0, count: num_confidence_scores - num_class_probs))
    let middleConf : [[Float]] = Array(repeating: innerMostConf, count: height)
    var pred_conf : [[[Float]]] = Array(repeating: middleConf, count: width)
    
    let innerMostBox : [Float] = Array<Float>(repeatElement(0.0, count: channels - num_confidence_scores))
    let middleBox : [[Float]] = Array(repeating: innerMostBox, count: height)
    var pred_box_delta : [[[Float]]] = Array(repeating: middleBox, count: width)
    
        for x in 0..<width{
            for y in 0..<height{
                pred_class_probs[x][y] = Math.softmax(Array(tensorflowImage[x][y][0..<num_class_probs]))
                for c in num_class_probs..<num_confidence_scores {
                    pred_conf[x][y][c-num_class_probs] = Math.sigmoid(tensorflowImage[x][y][c])
                }
                pred_box_delta[x][y] = Array(tensorflowImage[x][y][num_confidence_scores..<channels])
            }
        }
    
    
    
    
    
    //below old code_______________

    // This helper function finds the offset in the features array for a given
    // channel for a particular pixel. (See the comment below.)
    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
      let slice = channel / 4
      let indexInSlice = channel - slice*4
      let offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice
      return offset
    }

    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of 
    // five data items: x, y, width, height, and a confidence score. Each grid 
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell 
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 13x13x125 elements (actually x128 instead of x125 because in
    // Metal the number of channels must be a multiple of 4).

    for cy in 0..<gridHeight {
      for cx in 0..<gridWidth {
        for b in 0..<boxesPerCell {

          // The 13x13x125 image is arranged in planes of 4 channels. First are
          // channels 0-3 for the entire image, then channels 4-7 for the whole
          // image, then channels 8-11, and so on. Since we have 128 channels,
          // there are 128/4 = 32 of these planes (a.k.a. texture slices).
          //
          //    0123 0123 0123 ... 0123    ^
          //    0123 0123 0123 ... 0123    |
          //    0123 0123 0123 ... 0123    13 rows
          //    ...                        |
          //    0123 0123 0123 ... 0123    v
          //    4567 4557 4567 ... 4567
          //    etc
          //    <----- 13 columns ---->
          //
          // For the first bounding box (b=0) we have to read channels 0-24, 
          // for b=1 we have to read channels 25-49, and so on. Unfortunately,
          // these 25 channels are spread out over multiple slices. We use a
          // helper function to find the correct place in the features array.
          // (Note: It might be quicker / more convenient to transpose this
          // array so that all 125 channels are stored consecutively instead
          // of being scattered over multiple texture slices.)
          let channel = b*(numClasses + 5)
          let tx = features[offset(channel, cx, cy)]
          let ty = features[offset(channel + 1, cx, cy)]
          let tw = features[offset(channel + 2, cx, cy)]
          let th = features[offset(channel + 3, cx, cy)]
          let tc = features[offset(channel + 4, cx, cy)]

          // The predicted tx and ty coordinates are relative to the location 
          // of the grid cell; we use the logistic sigmoid to constrain these 
          // coordinates to the range 0 - 1. Then we add the cell coordinates 
          // (0-12) and multiply by the number of pixels per grid cell (32).
          // Now x and y represent center of the bounding box in the original
          // 416x416 image space.
          let x = (Float(cx) + Math.sigmoid(tx)) * blockSize
          let y = (Float(cy) + Math.sigmoid(ty)) * blockSize

          // The size of the bounding box, tw and th, is predicted relative to
          // the size of an "anchor" box. Here we also transform the width and
          // height into the original 416x416 image space.
          let w = exp(tw) * anchors[2*b    ] * blockSize
          let h = exp(th) * anchors[2*b + 1] * blockSize

          // The confidence value for the bounding box is given by tc. We use
          // the logistic sigmoid to turn this into a percentage.
          let confidence = Math.sigmoid(tc)

          // Gather the predicted classes for this anchor box and softmax them,
          // so we can interpret these numbers as percentages.
          var classes = [Float](repeating: 0, count: numClasses)
          for c in 0..<numClasses {
            classes[c] = features[offset(channel + 5 + c, cx, cy)]
          }
          classes = Math.softmax(classes)

          // Find the index of the class with the largest score.
          let (detectedClass, bestClassScore) = classes.argmax()

          // Combine the confidence score for the bounding box, which tells us
          // how likely it is that there is an object in this box (but not what
          // kind of object it is), with the largest class prediction, which
          // tells us what kind of object it detected (but not where).
          let confidenceInClass = bestClassScore * confidence

          // Since we compute 13x13x5 = 845 bounding boxes, we only want to
          // keep the ones whose combined score is over a certain threshold.
          if confidenceInClass > 0.3 {
            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                              width: CGFloat(w), height: CGFloat(h))

            let prediction = Prediction(classIndex: detectedClass,
                                        score: confidenceInClass,
                                        rect: rect)
            predictions.append(prediction)
          }
        }
      }
    }

    // We already filtered out any bounding boxes that have very low scores, 
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    var result = NeuralNetworkResult<Prediction>()
    result.predictions = nonMaxSuppression(boxes: predictions, limit: 10, threshold: 0.5)
    return result
  }
}

/**
  Removes bounding boxes that overlap too much with other boxes that have
  a higher score.
  
  Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

  - Parameters:
    - boxes: an array of bounding boxes and their scores
    - limit: the maximum number of boxes that will be selected
    - threshold: used to decide whether boxes overlap too much
*/
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

  // Do an argsort on the confidence scores, from high to low.
  let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

  var selected: [YOLO.Prediction] = []
  var active = [Bool](repeating: true, count: boxes.count)
  var numActive = active.count

  // The algorithm is simple: Start with the box that has the highest score. 
  // Remove any remaining boxes that overlap it more than the given threshold
  // amount. If there are any boxes left (i.e. these did not overlap with any
  // previous boxes), then repeat this procedure, until no more boxes remain 
  // or the limit has been reached.
  outer: for i in 0..<boxes.count {
    if active[i] {
      let boxA = boxes[sortedIndices[i]]
      selected.append(boxA)
      if selected.count >= limit { break }

      for j in i+1..<boxes.count {
        if active[j] {
          let boxB = boxes[sortedIndices[j]]
          if IOU(a: boxA.rect, b: boxB.rect) > threshold {
            active[j] = false
            numActive -= 1
            if numActive <= 0 { break outer }
          }
        }
      }
    }
  }
  return selected
}
