window.onload = function () {
    const image_input = document.querySelector("#image_input");
    const uploaded_image = document.querySelector("#img-place1");
    const be = document.getElementById("backend");
    const predictText = document.getElementById("predictText");

    // Hide predictText initially
    predictText.hidden = true;

    // Retrieve predicted bird species from localStorage on page load
    const predictedBird = localStorage.getItem('predictedBird');
    if (predictedBird !== null && predictedBird !== 'None') {
        // Display the predicted bird species if available
        predictText.innerText = predictedBird;
        predictText.hidden = false;
    }

    // Retrieve uploaded image data from localStorage on page load
    const uploadedImageData = localStorage.getItem('uploadedImageData');
    if (uploadedImageData !== null) {
        uploaded_image.style.backgroundImage = `url(${uploadedImageData})`;
    }

    image_input.addEventListener("change", function () {
        const reader = new FileReader();
        reader.addEventListener("load", () => {
            const imageData = reader.result;
            uploaded_image.style.backgroundImage = `url(${imageData})`;

            // Save the uploaded image data to localStorage
            localStorage.setItem('uploadedImageData', imageData);
        });
        reader.readAsDataURL(this.files[0]);
        remove();
        remove1();
    });

    function remove() {
        var element = document.getElementById("txt_main");
        element.classList.remove("txt");
        element.classList.add("txt5");
    }

    function remove1() {
        var elements = document.getElementById("pic");
        elements.classList.remove("icon1");
        elements.classList.add("plus-icon5");
    }

    // Function to fetch the predicted bird species from the server
    const fetchPredictedBird = async () => {
        try {
            const response = await fetch('http://localhost:8000/get_text');
            if (response.ok) {
                const data = await response.json();
                const newPredictedBird = data.predicted_name;
                // Check if the new predicted bird species is different
                if (newPredictedBird !== null && newPredictedBird !== 'None' && newPredictedBird !== predictedBird) {
                    // Update localStorage and display the new predicted bird species
                    localStorage.setItem('predictedBird', newPredictedBird);
                    predictText.innerText = newPredictedBird;
                    predictText.hidden = false;
                }
            } else {
                throw new Error('Error fetching predicted bird species');
            }
        } catch (error) {
            console.error('Error fetching predicted bird species:', error);
        }
    };

    // Fetch the predicted bird species every 2 seconds
    setInterval(fetchPredictedBird, 2000);

    be.addEventListener('click', async function submitImage() {
        const fileInput = document.getElementById('image_input');
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                console.log('Image successfully sent to the server');
                fetchPredictedBird();
            } else {
                throw new Error(`Error sending image to the server. Status: ${response.status}`);
            }
        } catch (error) {
            console.error('Error submitting image:', error);
        }
        return false;
    });

    document.getElementById('resetButton').addEventListener('click', async () => {
        try {
            const response = await fetch('http://localhost:8000/delete-image', {
                method: 'DELETE',
            });

            console.log(response.status);  // Log the response status

            if (response.ok) {
                console.log('Image deleted successfully');
                // Clear localStorage on image reset
                localStorage.removeItem('uploadedImageData');
            } else {
                throw new Error('Error deleting image');
            }
        } catch (error) {
            console.error('Error deleting image:', error);
        }
    });
};