// Dashboard Chart Initialization
document.addEventListener('DOMContentLoaded', function() {
    // Get metrics from Flask template
    const metrics = {
        accuracy: parseFloat('{{ metrics.accuracy|default(0.85, true) }}'),
        precision: parseFloat('{{ metrics.precision|default(0.82, true) }}'),
        recall: parseFloat('{{ metrics.recall|default(0.88, true) }}')
    };

    // Calculate F1 Score safely
    const f1Score = metrics.precision + metrics.recall > 0 
        ? (2 * metrics.precision * metrics.recall) / (metrics.precision + metrics.recall)
        : 0;

    // Performance Chart
    const perfCtx = document.getElementById('performanceChart');
    if (perfCtx) {
        new Chart(perfCtx, {
            type: 'bar',
            data: {
                labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                datasets: [{
                    label: 'Score',
                    data: [
                        metrics.accuracy,
                        metrics.precision,
                        metrics.recall,
                        f1Score
                    ],
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)',
                        'rgba(255, 159, 64, 0.7)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true, max: 1.0 } }
            }
        });
    }

    // Feature Importance Chart
    const featureCtx = document.getElementById('featureChart');
    if (featureCtx) {
        new Chart(featureCtx, {
            type: 'radar',
            data: {
                labels: ['Soil Moisture', 'Temperature', 'Humidity', 'Rainfall', 
                        'Wind Speed', 'Soil Type', 'Crop Type', 'Time'],
                datasets: [{
                    label: 'Importance',
                    data: [0.9, 0.7, 0.8, 0.4, 0.6, 0.3, 0.2, 0.5],
                    backgroundColor: 'rgba(255, 99, 132, 0.2)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                }]
            },
            options: {
                responsive: true,
                scales: { r: { suggestedMin: 0, suggestedMax: 1 } }
            }
        });
    }
});