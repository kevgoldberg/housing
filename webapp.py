import os
import json
from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route('/')
def index():
    return (
        '<h1>Housing Price Prediction Dashboard</h1>'
        '<ul>'
        '<li><a href="/metrics">Model Metrics</a></li>'
        '<li><a href="/visualizations">Visualizations</a></li>'
        '</ul>'
    )

@app.route('/metrics')
def metrics():
    if os.path.exists('metrics.json'):
        with open('metrics.json') as f:
            data = json.load(f)
        items = ''.join(f'<li>{k}: {v}</li>' for k, v in data.items())
        body = f'<h2>Model Metrics</h2><ul>{items}</ul>'
    else:
        body = '<p>No metrics available. Run training first.</p>'
    return body + '<p><a href="/">Back</a></p>'

@app.route('/visualizations')
def visualizations():
    viz_dir = 'visualizations'
    if not os.path.isdir(viz_dir):
        return '<p>No visualizations directory found.</p><p><a href="/">Back</a></p>'
    imgs = [f for f in os.listdir(viz_dir) if f.endswith('.png')]
    imgs.sort()
    tags = ''.join(f'<div><img src="/visualizations/{img}" style="max-width:600px;"></div>' for img in imgs)
    return f'<h2>Visualizations</h2>{tags}<p><a href="/">Back</a></p>'

@app.route('/visualizations/<path:filename>')
def viz_file(filename):
    return send_from_directory('visualizations', filename)

if __name__ == '__main__':
    app.run(debug=True)
