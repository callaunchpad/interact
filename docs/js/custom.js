/**
 * Declare config
 */
const config = {
    'api': 'https://interact.jimmyli.us/api/'
}

/**
 * jQuery to attach navbar for semantic
 */

$(document)
.ready(function() {

  // fix menu when passed
  $('.masthead')
    .visibility({
      once: false,
      onBottomPassed: function() {
        $('.fixed.menu').transition('fade in');
      },
      onBottomPassedReverse: function() {
        $('.fixed.menu').transition('fade out');
      }
    })
  ;

  // create sidebar and attach to menu open
  $('.ui.sidebar')
    .sidebar('attach events', '.toc.item')
  ;

});

const createParagraph = (text) => {
    const paragraphNode = document.createElement('p');
    paragraphNode.textContent = text;
    return paragraphNode;
}

window.onload = (_event) => {
    /**
     * HOCNN Form
     */
    const HOCNNDemoForm = document.getElementById('hocnn-demo');

    const submitHOCNNDemoForm = async (e) => {
        e.preventDefault();

        const formData = new FormData(HOCNNDemoForm);

        const predictions = await fetch(config.api + 'hocnn/predict', {
            'method': 'POST',
            'body': formData
        });
        const parsedPredictions = await predictions.json();
        const HOCNNResultsNode = document.getElementById('hocnn-results');
        HOCNNResultsNode.innerHTML = '';

        const parsedPredictionsKeys = Object.keys(parsedPredictions);
        for(let i = 0; i < parsedPredictionsKeys.length; i++){
            const predictionKey = parsedPredictionsKeys[i];
            if(predictionKey == 'fasterrcnn_object'){
              const predictionText = 'Faster-RCNN object: ' + parsedPredictions[predictionKey];
              HOCNNResultsNode.appendChild(createParagraph(predictionText));
            }else{
              const predictionText = '' + (i + 1) + ' ' + predictionKey + ' ' + parsedPredictions[predictionKey] + '%';
              HOCNNResultsNode.appendChild(createParagraph(predictionText));
            }
        }

        return false;
    }

    HOCNNDemoForm.addEventListener("submit", submitHOCNNDemoForm, false);

    /**
     * HOPOSECNN Form
     */
    const HOPOSECNNDemoForm = document.getElementById('hoposecnn-demo');

    const submitHOPOSECNNDemoForm = async (e) => {
        e.preventDefault();

        const formData = new FormData(HOPOSECNNDemoForm);

        const predictions = await fetch(config.api + 'hoposecnn/predict', {
            'method': 'POST',
            'body': formData
        });
        const parsedPredictions = await predictions.json();
        const HOPOSECNNResultsNode = document.getElementById('hoposecnn-results');
        HOPOSECNNResultsNode.innerHTML = '';

        const parsedPredictionsKeys = Object.keys(parsedPredictions);
        for(let i = 0; i < parsedPredictionsKeys.length; i++){
          const predictionKey = parsedPredictionsKeys[i];
          if(predictionKey == 'fasterrcnn_object'){
            const predictionText = 'Faster-RCNN object: ' + parsedPredictions[predictionKey];
            HOPOSECNNResultsNode.appendChild(createParagraph(predictionText));
          }else{
            const predictionText = '' + (i + 1) + ' ' + predictionKey + ' ' + parsedPredictions[predictionKey] + '%';
            HOPOSECNNResultsNode.appendChild(createParagraph(predictionText));
          }
      }

        return false;
    }

    HOPOSECNNDemoForm.addEventListener("submit", submitHOPOSECNNDemoForm, false);

    /**
     * Cool Background Net Form
     */
    const CBGNDemoForm = document.getElementById('cbgn-demo');

    const submitCBGNDemoForm = async (e) => {
        e.preventDefault();

        const formData = new FormData(CBGNDemoForm);

        const predictions = await fetch(config.api + 'cbgn/predict', {
            'method': 'POST',
            'body': formData
        });
        const parsedPredictions = await predictions.json();
        const CBGNResultsNode = document.getElementById('cbgn-results');
        CBGNResultsNode.innerHTML = '';

        const parsedPredictionsKeys = Object.keys(parsedPredictions);
        for(let i = 0; i < parsedPredictionsKeys.length; i++){
          const predictionKey = parsedPredictionsKeys[i];
          if(predictionKey == 'fasterrcnn_object'){
            const predictionText = 'Faster-RCNN object: ' + parsedPredictions[predictionKey];
            CBGNResultsNode.appendChild(createParagraph(predictionText));
          }else{
            const predictionText = '' + (i + 1) + ' ' + predictionKey + ' ' + parsedPredictions[predictionKey] + '%';
            CBGNResultsNode.appendChild(createParagraph(predictionText));
          }
      }

        return false;
    }

    CBGNDemoForm.addEventListener("submit", submitCBGNDemoForm, false);
};
