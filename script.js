const analyzeButton = document.getElementById("analyze-button");
const reviewTextarea = document.getElementById("review-text");
const resultValue = document.getElementById("result-value");

// Load the sentiment analysis ONNX model
let sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("./emotion_predictor.onnx");

// Initialize an empty object to hold the word-to-int dictionary
let wordToInt = {};

// Function to load the word_to_int dictionary from the JSON file
fetch('word_to_int.json')
    .then(response => response.json())
    .then(data => {
        wordToInt = data;
        console.log('Word to Int Dictionary Loaded:', wordToInt);
    })
    .catch(error => {
        console.error('Error loading word_to_int.json:', error);
    });

// Function to tokenize the input text (product review)
function encode(sentence) {
    sentence = sentence.replace(/[^a-zA-Z\s]/g, '').toLowerCase();
    return sentence.split(' ').map(word => wordToInt[word.toLowerCase()] || 0);  // Default to 0 if word is not found
}

// Function to handle review analysis
async function analyzeSentiment() {
  const review = reviewTextarea.value.trim();
  if (!review) {
    reviewTextarea.textContent = "Please enter a review!";
    updateSentimentScale(-2);
    return;
  }

  // Preprocess the text
  const processedText = encode(review); 

  // Convert the processed text into a tensor (this example assumes 1D tensor of token indices)
  const input = new onnx.Tensor(new Int32Array(processedText), "int32", [1, processedText.length]);
  
  sess = new onnx.InferenceSession();
  await sess.loadModel("./emotion_predictor.onnx");

  // Run the sentiment analysis model
  let outputMap;
  try { 
    outputMap = await sess.run([input]); 
  } catch { 
    try {
        outputMap = await sess.run([input]); 
    } catch {}
  }
  if (!outputMap) {
    updateSentimentScale(-2);
    return
  }
  const outputTensor = outputMap.values().next().value;
  const sentiment = outputTensor.data[0]; // Output is scalar sentiment value (e.g., -1 to 1 for sentiment strength)

  console.log("Sentiment Prediction (Tanh Output):", sentiment);
  updateSentimentScale(sentiment);
}

const resultScale = document.getElementById('result-scale');

function updateSentimentScale(sentiment) {
    // Calculate the position of the scale from -1 (negative) to 1 (positive)
    let scaleValue = (sentiment + 1) / 2;  // Normalize to range from 0 to 1
    if (scaleValue < 0) {
        scaleValue = 0
    }
    if (scaleValue < 0.05) {
        scaleValue = 0.05
    }
    // Set the scale width based on the sentiment score (between 0 and 100%)
    resultScale.style.width = `${scaleValue * 100}%`;
  
    // Optionally change the color of the scale based on sentiment
    if (sentiment > 0.1) {
      resultScale.style.backgroundColor = "#2ecc71";  // Green for positive sentiment
    } else if (sentiment < -0.1) {
      resultScale.style.backgroundColor = "#e74c3c";  // Red for negative sentiment
    } else {
      resultScale.style.backgroundColor = "#f39c12";  // Yellow for neutral sentiment
    }
  }

// Add event listener to analyze the review
loadingModelPromise.then(() => {
  reviewTextarea.addEventListener("input", analyzeSentiment);
});
