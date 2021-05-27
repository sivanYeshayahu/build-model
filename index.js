const classifier = knnClassifier.create(); //creat new classifier
// classifier = JSON.parse(our file in hard drive)
const webcamElement = document.getElementById('webcam');


// let SIVAN={
//     name: "S",
//     family: "Y"
// }
//
// let example = JSON.stringlify(SIVAN)
//
// os.save(path, example)

let net;

async function app() {
    console.log('Loading mobilenet..');


    // Load the model.
    net = await mobilenet.load();
    console.log('Successfully loaded model');


    const imgsUpload = document.getElementById('uploadImages');

    imgsUpload.addEventListener("change", handleFiles);
    function handleFiles() {

        for (let i = 0; i < this.files.length; i++) {
            const file = this.files[i];

            const img = document.createElement("img");
            img.classList.add("obj");
            img.file = file;

            const reader = new FileReader();
            reader.onload = (function (aImg) {
                return function (e) {
                    aImg.src = e.target.result;
                };
            })(img);
            reader.readAsDataURL(file);
            addExample(0, img);

        }
    }


    // Create an object from Tensorflow.js data API which could capture image
    // from the web camera as Tensor.
    const webcam = await tf.data.webcam(webcamElement);

    // Reads an image from the webcam and associates it with a specific class
    // index.
    const addExample = async (classId, img) => {//
        // Capture an image from the web camera.
        if(!img){
            img = await webcam.capture();
        }
        else
        {
              img= tf.node.decodeJpeg(img)
            //const saveResult = await model.save('localstorage://my-model-1');//////////////////////////////////
        }


        // Get the intermediate activation of MobileNet 'conv_preds' and pass that
        // to the KNN classifier.
        const activation = net.infer(img, true);

        // Pass the intermediate activation to the classifier.
        classifier.addExample(activation, classId);  //add image

        //model= JSON.stringlify(classifier)    ///////// to save model on hard drive


        // Dispose the tensor to release the memory.
        img.dispose();
    };

    // When clicking a button, add an example for that class.
    document.getElementById('class-a').addEventListener('click', () => addExample(0));
    document.getElementById('class-b').addEventListener('click', () => addExample(1));
    document.getElementById('class-c').addEventListener('click', () => addExample(2));
    document.getElementById('class-d').addEventListener('click', () => addExample(3));

    while (true) {
        if (classifier.getNumClasses() > 0) {   ////////// machine learning
            const img = await webcam.capture();//open web cam

            // Get the activation from mobilenet from the webcam.
            const activation = net.infer(img, 'conv_preds');
            // Get the most likely class and confidence from the classifier module.
            const result = await classifier.predictClass(activation);


            const classes = ['A', 'B', 'C', 'D'];
            document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

            // Dispose the tensor to release the memory.
            img.dispose();
        }
        // Give some breathing room by waiting for the next animation frame to
        // fire.
        await tf.nextFrame();
    }

//     saveButton = createButton('save');
//   saveButton.mousePressed(function() {
//     classifier.save();
//   }); //


}

app();