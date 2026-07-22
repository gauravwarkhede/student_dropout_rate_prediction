import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'LogisticRegression.pkl')

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
        
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
    else:
        feature_names = [
            'Student_ID', 'Age', 'Gender', 'Family_Income', 'Internet_Access', 
            'Study_Hours_per_Day', 'Attendance_Rate', 'Assignment_Delay_Days', 
            'Travel_Time_Minutes', 'Part_Time_Job', 'Scholarship', 'Stress_Index', 
            'GPA', 'Semester_GPA', 'CGPA', 'Semester', 'Department', 'Parental_Education'
        ]
        
    feature_importance = model.coef_[0].tolist() if hasattr(model, 'coef_') else [0] * len(feature_names)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []
    feature_importance = []

# --- HTML/UI TEMPLATE ---
# The entire glassmorphism dashboard is stored here to keep it in one file.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Dropout Analytics Engine</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {
            --bg-color: #0b1120;
            --surface-glass: rgba(30, 41, 59, 0.7);
            --surface-border: rgba(255, 255, 255, 0.1);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #3b82f6; 
            --accent-secondary: #14b8a6; 
            --danger: #ef4444;
            --success: #10b981;
            --font-family: 'Inter', sans-serif;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-family);
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(20, 184, 166, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }
        .container {
            width: 95%;
            max-width: 1400px;
            margin: 2rem auto;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
        }
        .glass-card {
            background: var(--surface-glass);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--surface-border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover { transform: translateY(-5px); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2); }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.025em; }
        .app-header { text-align: center; padding: 3rem 1rem 1rem 1rem; animation: fadeInDown 0.8s ease-out; }
        .app-header h1 { font-size: 2.5rem; background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .input-group { margin-bottom: 1.2rem; }
        .input-group label { display: block; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .input-group input { width: 100%; padding: 0.8rem 1rem; background: rgba(15, 23, 42, 0.6); border: 1px solid var(--surface-border); border-radius: 8px; color: var(--text-primary); font-size: 1rem; transition: all 0.3s ease; }
        .input-group input:focus { outline: none; border-color: var(--accent-primary); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2); }
        .form-container { max-height: 75vh; overflow-y: auto; padding-right: 10px; }
        .form-container::-webkit-scrollbar { width: 6px; }
        .form-container::-webkit-scrollbar-track { background: transparent; }
        .form-container::-webkit-scrollbar-thumb { background: var(--surface-border); border-radius: 4px; }
        .btn-predict { width: 100%; padding: 1rem; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); border: none; border-radius: 8px; color: white; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 0.1em; position: relative; }
        .btn-predict:hover { filter: brightness(1.1); transform: scale(1.02); }
        .dashboard-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }
        .full-width { grid-column: 1 / -1; }
        .result-display { text-align: center; padding: 2rem; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .result-status { font-size: 2rem; font-weight: bold; margin-top: 1rem; transition: color 0.5s ease; }
        .status-danger { color: var(--danger); }
        .status-success { color: var(--success); }
        .metric-cards { display: flex; justify-content: space-around; width: 100%; margin-top: 1.5rem; }
        .metric { text-align: center; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; min-width: 120px; }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: var(--accent-secondary); }
        .metric-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; }
        .chart-container { position: relative; height: 300px; width: 100%; }
        .radar-container { position: relative; height: 350px; width: 100%; }
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .animate-up { animation: fadeInUp 0.8s ease-out forwards; opacity: 0; }
        .delay-1 { animation-delay: 0.2s; }
        .delay-2 { animation-delay: 0.4s; }
        @media (max-width: 968px) { .container, .dashboard-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <header class="app-header">
        <h1>Predictive Analytics Engine</h1>
        <p style="color: var(--text-secondary); margin-top: 0.5rem;">Advanced Logistic Regression Model</p>
    </header>

    <div class="container">
        <aside class="glass-card animate-up delay-1">
            <h2>Student Profile Input</h2>
            <div class="form-container">
                <form id="predictionForm">
                    <script>
                        const formFeatures = [
                            {id: 'Student_ID', label: 'Student ID', type: 'number', step: '1', val: 1001},
                            {id: 'Age', label: 'Age', type: 'number', step: '1', val: 20},
                            {id: 'Gender', label: 'Gender (0=M, 1=F)', type: 'number', step: '1', val: 1},
                            {id: 'Family_Income', label: 'Family Income ($)', type: 'number', step: '1000', val: 45000},
                            {id: 'Internet_Access', label: 'Internet Access (0=No, 1=Yes)', type: 'number', step: '1', val: 1},
                            {id: 'Study_Hours_per_Day', label: 'Study Hours/Day', type: 'number', step: '0.1', val: 3.5},
                            {id: 'Attendance_Rate', label: 'Attendance Rate (%)', type: 'number', step: '0.1', val: 85.0},
                            {id: 'Assignment_Delay_Days', label: 'Avg Assignment Delay', type: 'number', step: '1', val: 2},
                            {id: 'Travel_Time_Minutes', label: 'Travel Time (Mins)', type: 'number', step: '1', val: 30},
                            {id: 'Part_Time_Job', label: 'Part Time Job (0=No, 1=Yes)', type: 'number', step: '1', val: 0},
                            {id: 'Scholarship', label: 'Scholarship (0=No, 1=Yes)', type: 'number', step: '1', val: 1},
                            {id: 'Stress_Index', label: 'Stress Index (1-10)', type: 'number', step: '0.1', val: 5.5},
                            {id: 'GPA', label: 'Current GPA', type: 'number', step: '0.01', val: 3.2},
                            {id: 'Semester_GPA', label: 'Last Semester GPA', type: 'number', step: '0.01', val: 3.1},
                            {id: 'CGPA', label: 'Cumulative GPA', type: 'number', step: '0.01', val: 3.15},
                            {id: 'Semester', label: 'Current Semester', type: 'number', step: '1', val: 4},
                            {id: 'Department', label: 'Department Code', type: 'number', step: '1', val: 2},
                            {id: 'Parental_Education', label: 'Parental Education', type: 'number', step: '1', val: 3}
                        ];
                        formFeatures.forEach(f => {
                            document.write(`
                                <div class="input-group">
                                    <label for="${f.id}">${f.label}</label>
                                    <input type="${f.type}" id="${f.id}" name="${f.id}" step="${f.step}" value="${f.val}" required>
                                </div>
                            `);
                        });
                    </script>
                    <button type="submit" class="btn-predict" id="submitBtn">Run Analysis</button>
                </form>
            </div>
        </aside>

        <main class="dashboard-grid animate-up delay-2">
            <div class="glass-card full-width result-display">
                <h3 style="color: var(--text-secondary);">Model Inference Result</h3>
                <div id="resultOutput" class="result-status">Awaiting Input...</div>
                <div class="metric-cards">
                    <div class="metric"><div id="probStay" class="metric-value">--%</div><div class="metric-label">Retention Prob.</div></div>
                    <div class="metric"><div id="probDrop" class="metric-value">--%</div><div class="metric-label">Dropout Prob.</div></div>
                    <div class="metric"><div id="riskLevel" class="metric-value" style="color: #fff;">--</div><div class="metric-label">Assessed Risk</div></div>
                </div>
            </div>

            <div class="glass-card">
                <h3>Prediction Confidence</h3>
                <div class="chart-container"><canvas id="donutChart"></canvas></div>
            </div>

            <div class="glass-card">
                <h3>Student Behavioral Profile</h3>
                <div class="radar-container"><canvas id="radarChart"></canvas></div>
            </div>

            <div class="glass-card full-width">
                <h3>Logistic Regression Weights</h3>
                <div class="chart-container" style="height: 350px;"><canvas id="barChart"></canvas></div>
            </div>
        </main>
    </div>

    <script>
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
        Chart.defaults.plugins.tooltip.titleColor = '#fff';
        
        let donutChart, radarChart, barChart;
        const globalFeatures = {{ features | tojson | safe }};
        const globalImportances = {{ importance | tojson | safe }};

        function initCharts() {
            const ctxDonut = document.getElementById('donutChart').getContext('2d');
            donutChart = new Chart(ctxDonut, {
                type: 'doughnut',
                data: { labels: ['Will Not Dropout', 'Will Dropout'], datasets: [{ data: [50, 50], backgroundColor: ['#10b981', '#ef4444'], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'bottom' } } }
            });

            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Attendance', 'GPA', 'Study Hours', 'Stress', 'Delay'],
                    datasets: [{ label: 'Current Student', data: [0,0,0,0,0], backgroundColor: 'rgba(59, 130, 246, 0.2)', borderColor: '#3b82f6', pointBackgroundColor: '#14b8a6', borderWidth: 2 }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { r: { angleLines: { color: 'rgba(255, 255, 255, 0.1)' }, grid: { color: 'rgba(255, 255, 255, 0.1)' }, pointLabels: { color: '#94a3b8' }, ticks: { display: false } } } }
            });

            const bgColors = globalImportances.map(val => val < 0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)');
            const borderColors = globalImportances.map(val => val < 0 ? '#10b981' : '#ef4444');
            const ctxBar = document.getElementById('barChart').getContext('2d');
            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: { labels: globalFeatures, datasets: [{ label: 'Coefficient Impact', data: globalImportances, backgroundColor: bgColors, borderColor: borderColors, borderWidth: 1, borderRadius: 4 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }, x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } } }, plugins: { legend: { display: false } } }
            });
        }

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const dataObj = {};
            formData.forEach((value, key) => { dataObj[key] = parseFloat(value); });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(dataObj)
                });
                const result = await response.json();
                if(response.ok) { updateDashboard(result, dataObj); } else { alert('Error: ' + result.error); }
            } catch (error) { alert('Connection error: ' + error.message); }
        });

        function updateDashboard(apiData, inputData) {
            const resOut = document.getElementById('resultOutput');
            resOut.textContent = apiData.prediction; 
            resOut.className = apiData.prediction_code === 1 ? 'result-status status-danger' : 'result-status status-success';
            
            document.getElementById('probStay').textContent = apiData.probability_stay + '%';
            document.getElementById('probDrop').textContent = apiData.probability_dropout + '%';
            
            const riskSpan = document.getElementById('riskLevel');
            riskSpan.textContent = apiData.risk_level;
            riskSpan.style.color = apiData.risk_level === 'High' ? '#ef4444' : '#10b981';

            donutChart.data.datasets[0].data = [apiData.probability_stay, apiData.probability_dropout];
            donutChart.update();

            radarChart.data.datasets[0].data = [
                (inputData.Attendance_Rate || 0) / 100 * 100,
                ((inputData.GPA || 0) / 4.0) * 100,
                ((inputData.Study_Hours_per_Day || 0) / 12) * 100,
                ((inputData.Stress_Index || 0) / 10) * 100,
                (20 - (inputData.Assignment_Delay_Days || 0)) / 20 * 100 
            ];
            radarChart.update();
        }

        window.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
"""

# --- ROUTES ---
@app.route('/')
def home():
    """Renders the inline HTML template."""
    return render_template_string(
        HTML_TEMPLATE, 
        features=feature_names,
        importance=feature_importance
    )

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the model inference."""
    if not model:
        return jsonify({'error': 'Model failed to load.'}), 500

    try:
        data = request.get_json()
        
        input_data = []
        for feature in feature_names:
            val = float(data.get(feature, 0.0))
            input_data.append(val)
            
        features_array = np.array(input_data).reshape(1, -1)
        
        prediction_val = int(model.predict(features_array)[0])
        probabilities = model.predict_proba(features_array)[0]
        
        if prediction_val == 1:
            result_text = "Student will dropout"
            risk_level = "High"
        else:
            result_text = "Student will not dropout"
            risk_level = "Low"
            
        return jsonify({
            'prediction': result_text,
            'prediction_code': prediction_val,
            'probability_dropout': round(probabilities[1] * 100, 2),
            'probability_stay': round(probabilities[0] * 100, 2),
            'risk_level': risk_level,
            'input_echo': input_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run locally
    app.run(debug=True, port=5000)
