async function startRetraining() {
    const fileInput = document.getElementById('dataset');
    const epochs = document.getElementById('epochs').value;
    const batchSize = document.getElementById('batch_size').value;

    if (!fileInput.files[0]) {
        alert("Please select a file first!");
        return;
    }

    // Step 1: Upload the file
    const formData = new FormData();
    formData.append('dataset', fileInput.files[0]);
    formData.append('epochs', epochs);
    formData.append('batch_size', batchSize);

    try {
        const uploadResponse = await fetch('/retrain', {
            method: 'POST',
            body: formData,
        });

        const uploadResult = await uploadResponse.json();

        if (uploadResult.status !== 'success') {
            alert('File upload failed: ' + uploadResult.message);
            return;
        }

        // Step 2: Start retraining stream with the received filename
        const eventSource = new EventSource(
            `/retrain/stream?filename=${uploadResult.filename}&epochs=${uploadResult.epochs}&batch_size=${uploadResult.batch_size}`
        );

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data);

            if (data.status === 'completed') {
                eventSource.close();
                alert(`Retraining completed! Accuracy: ${data.accuracy}`);
            } else if (data.status === 'error') {
                eventSource.close();
                alert(`Error: ${data.message}`);
            }
        };

        eventSource.onerror = () => {
            eventSource.close();
            alert("Retraining stream failed.");
        };
    } catch (error) {
        alert("Error: " + error.message);
    }
}

// Attach the function to your retrain button
document.getElementById('retrain-button').addEventListener('click', startRetraining);