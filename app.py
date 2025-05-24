from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

#load file default untuk fitur via_app
df_default = pd.read_csv("social_media_usage.csv")
df_default.columns = df_default.columns.str.strip()

# Untuk fitur via_csv
df_uploaded = None
df_clustered = None

# Utility function to generate visualizations
def generate_visualizations(df):
    global visuals_generated
    features = ["Daily_Minutes_Spent", "Posts_Per_Day", "Likes_Per_Day", "Follows_Per_Day"]
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    plt.figure()
    plt.plot(range(1, 10), inertia, marker='o')
    plt.title('Metode Elbow')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Inertia')
    plt.savefig('static/elbow.png')

    # Silhouette Score
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    plt.figure()
    plt.plot(range(2, 10), silhouette_scores, marker='s', color='orange')
    plt.title('Silhouette Score')
    plt.xlabel('Jumlah Cluster')
    plt.ylabel('Silhouette Score')
    plt.savefig('static/silhouette.png')

    # Final Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    plt.figure()
    sns.scatterplot(x=df['Daily_Minutes_Spent'], y=df['Posts_Per_Day'], hue=df['Cluster'], palette='viridis')
    plt.title('Hasil Clustering')
    plt.savefig('static/cluster_plot.png')
    visuals_generated = True
    return df

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    global df_uploaded, df_clustered
    if request.method == 'POST':
        file = request.files['csv_file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            df_uploaded = pd.read_csv(filepath)
            df_uploaded.columns = df_uploaded.columns.str.strip()
            df_clustered = generate_visualizations(df_uploaded.copy())
            return redirect(url_for('data_awal'))
    return render_template('upload_csv.html')

@app.route('/data_awal')
def data_awal():
    if df_uploaded is None:
        return redirect(url_for('upload_csv'))
    data = df_default.to_html(classes='table table-bordered', index=False)
    return render_template('data_awal.html', table=data)

@app.route('/hasil_clustering')
def hasil_clustering():
    if df_clustered is None:
        return redirect(url_for('upload_csv'))
    data = df_clustered.to_html(classes='table table-bordered', index=False)
    return render_template('hasil_clustering.html', table=data)

@app.route('/visualisasi')
def visualisasi():
    return render_template('visualisasi.html')

@app.route('/input', methods=['GET', 'POST'])
def input():
    if request.method == 'POST':
        app_name = request.form.get('app', '').strip()
        return redirect(url_for('via_app', app_name=app_name))
    return render_template('input.html')

@app.route('/via_app')
def via_app():
    global df_default
    app_name = request.args.get('app_name', '').strip()
    result_table = ""

    if df_default is None or 'App' not in df_default.columns:
        return "Data belum diupload atau kolom 'App' tidak tersedia.", 400

    df_default.columns = df_default.columns.str.strip()
    filtered_df = df_default[df_default['App'].str.lower() == app_name.lower()]
    if not filtered_df.empty:
        df_temp = generate_visualizations(filtered_df.copy())
        result_table = df_temp.to_html(classes='table table-bordered', index=False)
    else:
        result_table = f"<p>Tidak ada data ditemukan untuk app: <strong>{app_name}</strong></p>"

    return render_template('via_app.html', table=result_table, app_name=app_name)

if __name__ == '__main__':
    app.run(debug=True)
