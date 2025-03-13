from app import app

if __name__ == "__main__":
    print("Running the Flask app via predict_api.py")
    app.run(debug=True, port=5001)
