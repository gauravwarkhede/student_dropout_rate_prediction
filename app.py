# ==========================================
# FILE 1: requirements.txt (Required by Vercel)
# Create a separate file named requirements.txt and paste this:
# 
# Flask==3.0.2
# Werkzeug==3.0.1
# scikit-learn==1.6.1
# numpy==1.26.4
# pandas==2.2.0
# scipy==1.12.0
# ==========================================

# ==========================================
# FILE 2: vercel.json (Required by Vercel)
# Create a separate file named vercel.json and paste this:
# 
# {
#   "version": 2,
#   "builds": [{"src": "app.py", "use": "@vercel/python"}],
#   "routes": [{"src": "/(.*)", "dest": "app.py"}]
# }
# ==========================================

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
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Dropout Analytics Engine</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        /* --- THEME DEFINITIONS --- */
        :root {
            /* Default: Midnight Glass */
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
            --bg-gradient: radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(20, 184, 166, 0.1) 0px, transparent 50%);
        }

        body.theme-cyberpunk {
            --bg-color: #050510;
            --surface-glass: rgba(20, 5, 35, 0.75);
            --surface-border: rgba(255, 0, 124, 0.2);
            --accent-primary: #ff007c; 
            --accent-secondary: #00f0ff; 
            --danger: #ff3333;
            --success: #00ff9d;
            --bg-gradient: radial-gradient(at 0% 0%, rgba(255, 0, 124, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(0, 240, 255, 0.1) 0px, transparent 50%);
        }

        body.theme-forest {
            --bg-color: #07130c;
            --surface-glass: rgba(15, 30, 20, 0.7);
            --surface-border: rgba(46, 160, 67, 0.2);
            --text-secondary: #a3b8a8;
            --accent-primary: #2ea043; 
            --accent-secondary: #a8b545; 
            --danger: #d73a49;
            --success: #28a745;
            --bg-gradient: radial-gradient(at 0% 0%, rgba(46, 160, 67, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(168, 181, 69, 0.1) 0px, transparent 50%);
        }

        body.theme-ice {
            --bg-color: #000914;
            --surface-glass: rgba(10, 25, 45, 0.7);
            --surface-border: rgba(56, 189, 248, 0.2);
            --text-secondary: #bae6fd;
            --accent-primary: #38bdf8; 
            --accent-secondary: #c084fc; 
            --danger: #f43f5e;
            --success: #0ea5e9;
            --bg-gradient: radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(192, 132, 252, 0.1) 0px, transparent 50%);
        }

        body.theme-magic {
            --bg-color: #11061f;
            --surface-glass: rgba(35, 15, 55, 0.7);
            --surface-border: rgba(168, 85, 247, 0.2);
            --text-secondary: #e9d5ff;
            --accent-primary: #a855f7; 
            --accent-secondary: #ec4899; 
            --danger: #f43f5e;
            --success: #d946ef;
            --bg-gradient: radial-gradient(at 0% 0%, rgba(168, 85, 247, 0.15) 0px, transparent 50%),
                           radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 50%);
        }

        /* --- GLOBAL STYLES --- */
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: var(--font-family);
            background-color: var(--bg-color);
            background-image: var(--bg-gradient);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            transition: background-color 0.5s ease, background-image 0.5s ease;
        }

        /* Top Bar & Theme Switcher */
        .top-nav {
            display: flex;
            justify-content: flex-end;
            padding: 1rem 2rem 0;
        }
        .theme-selector {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--surface-border);
            color: var(--text-primary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-family: var(--font-family);
            outline: none;
            cursor: pointer;
            transition: border-color 0.3s ease;
        }
        .theme-selector:focus { border-color: var(--accent-primary); }
        .theme-selector option { background: #000; color: #fff; }

        /* Layout */
        .container {
            width: 95%;
            max-width: 1400px;
            margin: 0 auto 2rem auto;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
        }

        /* Cards */
        .glass-card {
            background: var(--surface-glass);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--surface-border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.5s ease, border-color 0.5s ease;
        }
        .glass-card:hover { transform: translateY(-5px); box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4); }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.025em; }
        
        .app-header { text-align: center; padding: 1rem 1rem 2rem 1rem; animation: fadeInDown 0.8s ease-out; }
        .app-header h1 { font-size: 2.5rem; background: linear-gradient(to right, var(--accent-primary), var(--accent-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; transition: all 0.5s ease; }
        
        /* Inputs */
        .input-group { margin-bottom: 1.2rem; }
        .input-group label { display: block; font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; transition: color 0.5s ease; }
        .input-group input { width: 100%; padding: 0.8rem 1rem; background: rgba(0, 0, 0, 0.4); border: 1px solid var(--surface-border); border-radius: 8px; color: var(--text-primary); font-size: 1rem; transition: all 0.3s ease; }
        .input-group input:focus { outline: none; border-color: var(--accent-primary); box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.1); }
        
        .form-container { max-height: 75vh; overflow-y: auto; padding-right: 10px; }
        .form-container::-webkit-scrollbar { width: 6px; }
        .form-container::-webkit-scrollbar-track { background: transparent; }
        .form-container::-webkit-scrollbar-thumb { background: var(--surface-border); border-radius: 4px; }
        
        /* Buttons */
        .btn-predict { width: 100%; padding: 1rem; background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%); border: none; border-radius: 8px; color: white; font-size: 1.1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease; text-transform: uppercase; letter-spacing: 0.1em; position: relative; }
        .btn-predict:hover { filter: brightness(1.2); transform: scale(1.02); }
        
        /* Dashboard */
        .dashboard-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }
        .full-width { grid-column: 1 / -1; }
        .result-display { text-align: center; padding: 2rem; display: flex; flex-direction: column; justify-content: center; align-items: center; }
        .result-status { font-size: 2rem; font-weight: bold; margin-top: 1rem; transition: color 0.5s ease; }
        .status-danger { color: var(--danger); }
        .status-success { color: var(--success); }
        
        /* Metrics */
        .metric-cards { display: flex; justify-content: space-around; width: 100%; margin-top: 1.5rem; }
        .metric { text-align: center; padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 8px; min-width: 120px; border: 1px solid transparent; transition: border-color 0.5s ease; }
        .metric:hover { border-color: var(--surface-border); }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: var(--accent-secondary); transition: color 0.5s ease;}
        .metric-label { font-size: 0.75rem; color: var(--text-secondary); text-transform: uppercase; transition: color 0.5s ease;}
        
        .chart-container { position: relative; height: 300px; width: 100%; }
        .radar-container { position: relative; height: 350px; width: 100%; }
        
        /* Animations */
        @keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .animate-up { animation: fadeInUp 0.8s ease-out forwards; opacity: 0; }
        .delay-1 { animation-delay: 0.2s; }
        .delay-2 { animation-delay: 0.4s; }
        
        @media (max-width: 968px) { .container, .dashboard-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body class="theme-default">

    <!-- Top Navigation / Theme Selector -->
    <nav class="top-nav animate-up">
        <select id="themeSelector" class="theme-selector" onchange="changeTheme(this.value)">
            <option value="theme-default">Theme: Midnight Glass</option>
            <option value="theme-cyberpunk">Theme: Cyberpunk</option>
            <option value="theme-forest">Theme: Deep Forest</option>
            <option value="theme-ice">Theme: Glacial Ice</option>
            <option value="theme-magic">Theme: Magic Purple</option>
        </select>
    </nav>

    <header class="app-header">
        <h1>Predictive Analytics Engine</h1>
        <p style="color: var(--text-secondary); margin-top: 0.5rem; transition: color 0.5s ease;">Advanced Logistic Regression Model</p>
    </header>

    <div class="container">
        <!-- Sidebar Input Form -->
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

        <!-- Analytics Dashboard -->
        <main class="dashboard-grid animate-up delay-2">
            <div class="glass-card full-width result-display">
                <h3 style="color: var(--text-secondary); transition: color 0.5s ease;">Model Inference Result</h3>
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
        // --- DATA SETUP ---
        const globalFeatures = {{ features | tojson | safe }};
        const globalImportances = {{ importance | tojson | safe }};
        let donutChart, radarChart, barChart;

        // --- THEME COLOR ENGINE ---
        // Maps body class names to specific hex codes for Chart.js redrawing
        const themePalette = {
            'theme-default': { primary: '#3b82f6', secondary: '#14b8a6', danger: '#ef4444', success: '#10b981', grid: 'rgba(255,255,255,0.1)' },
            'theme-cyberpunk': { primary: '#ff007c', secondary: '#00f0ff', danger: '#ff3333', success: '#00ff9d', grid: 'rgba(255,0,124,0.2)' },
            'theme-forest': { primary: '#2ea043', secondary: '#a8b545', danger: '#d73a49', success: '#28a745', grid: 'rgba(46,160,67,0.2)' },
            'theme-ice': { primary: '#38bdf8', secondary: '#c084fc', danger: '#f43f5e', success: '#0ea5e9', grid: 'rgba(56,189,248,0.2)' },
            'theme-magic': { primary: '#a855f7', secondary: '#ec4899', danger: '#f43f5e', success: '#d946ef', grid: 'rgba(168,85,247,0.2)' }
        };

        // Convert hex to rgba for radar backgrounds
        function hexToRgba(hex, alpha) {
            let r = parseInt(hex.slice(1, 3), 16),
                g = parseInt(hex.slice(3, 5), 16),
                b = parseInt(hex.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }

        // --- CHART INITIALIZATION ---
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.font.family = "'Inter', sans-serif";
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
        Chart.defaults.plugins.tooltip.titleColor = '#fff';

        function initCharts() {
            const currentTheme = themePalette['theme-default'];

            // Donut Chart
            const ctxDonut = document.getElementById('donutChart').getContext('2d');
            donutChart = new Chart(ctxDonut, {
                type: 'doughnut',
                data: { labels: ['Will Not Dropout', 'Will Dropout'], datasets: [{ data: [50, 50], backgroundColor: [currentTheme.success, currentTheme.danger], borderWidth: 0 }] },
                options: { responsive: true, maintainAspectRatio: false, cutout: '75%', plugins: { legend: { position: 'bottom' } }, animation: { duration: 1000 } }
            });

            // Radar Chart
            const ctxRadar = document.getElementById('radarChart').getContext('2d');
            radarChart = new Chart(ctxRadar, {
                type: 'radar',
                data: {
                    labels: ['Attendance', 'GPA', 'Study Hours', 'Stress', 'Delay'],
                    datasets: [{ label: 'Current Student', data: [0,0,0,0,0], backgroundColor: hexToRgba(currentTheme.primary, 0.2), borderColor: currentTheme.primary, pointBackgroundColor: currentTheme.secondary, borderWidth: 2 }]
                },
                options: { responsive: true, maintainAspectRatio: false, scales: { r: { angleLines: { color: currentTheme.grid }, grid: { color: currentTheme.grid }, pointLabels: { color: '#94a3b8' }, ticks: { display: false } } } }
            });

            // Bar Chart (Feature Importance)
            const bgColors = globalImportances.map(val => val < 0 ? hexToRgba(currentTheme.success, 0.7) : hexToRgba(currentTheme.danger, 0.7));
            const borderColors = globalImportances.map(val => val < 0 ? currentTheme.success : currentTheme.danger);
            const ctxBar = document.getElementById('barChart').getContext('2d');
            barChart = new Chart(ctxBar, {
                type: 'bar',
                data: { labels: globalFeatures, datasets: [{ label: 'Coefficient Impact', data: globalImportances, backgroundColor: bgColors, borderColor: borderColors, borderWidth: 1, borderRadius: 4 }] },
                options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }, x: { grid: { display: false }, ticks: { maxRotation: 45, minRotation: 45 } } }, plugins: { legend: { display: false } } }
            });
        }

        // --- THEME SWITCHER LOGIC ---
        function changeTheme(themeClass) {
            // Update Body Class
            document.body.className = themeClass;
            const colors = themePalette[themeClass];

            // Update Donut Chart
            donutChart.data.datasets[0].backgroundColor = [colors.success, colors.danger];
            donutChart.update();

            // Update Radar Chart
            radarChart.data.datasets[0].backgroundColor = hexToRgba(colors.primary, 0.2);
            radarChart.data.datasets[0].borderColor = colors.primary;
            radarChart.data.datasets[0].pointBackgroundColor = colors.secondary;
            radarChart.options.scales.r.angleLines.color = colors.grid;
            radarChart.options.scales.r.grid.color = colors.grid;
            radarChart.update();

            // Update Bar Chart
            const newBgColors = globalImportances.map(val => val < 0 ? hexToRgba(colors.success, 0.7) : hexToRgba(colors.danger, 0.7));
            const newBorderColors = globalImportances.map(val => val < 0 ? colors.success : colors.danger);
            barChart.data.datasets[0].backgroundColor = newBgColors;
            barChart.data.datasets[0].borderColor = newBorderColors;
            barChart.update();
            
            // Force redraw risk text if it's already generated
            const riskSpan = document.getElementById('riskLevel');
            if(riskSpan.textContent === 'High') { riskSpan.style.color = colors.danger; }
            else if(riskSpan.textContent === 'Low') { riskSpan.style.color = colors.success; }
        }

        // --- FORM SUBMISSION & API CALL ---
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

        // --- DASHBOARD UPDATER ---
        function updateDashboard(apiData, inputData) {
            const currentTheme = document.body.className;
            const colors = themePalette[currentTheme];

            const resOut = document.getElementById('resultOutput');
            resOut.textContent = apiData.prediction; 
            
            if (apiData.prediction_code === 1) {
                resOut.className = 'result-status'; // Reset
                resOut.style.color = colors.danger;
            } else {
                resOut.className = 'result-status'; // Reset
                resOut.style.color = colors.success;
            }
            
            document.getElementById('probStay').textContent = apiData.probability_stay + '%';
            document.getElementById('probDrop').textContent = apiData.probability_dropout + '%';
            
            const riskSpan = document.getElementById('riskLevel');
            riskSpan.textContent = apiData.risk_level;
            riskSpan.style.color = apiData.risk_level === 'High' ? colors.danger : colors.success;

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
