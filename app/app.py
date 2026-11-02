import os
import sys
import json
from datetime import datetime, timezone
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

# Ensure modules in current and parent directory can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import db, User, PredictionHistory
from predictor import Predictor, predict_news

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(32)
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.login_view = 'login'
    login_manager.login_message_category = 'warning'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Pre-load LSTM model and tokenizer into memory once at app startup
    with app.app_context():
        db.create_all()
        try:
            print("[App Startup] Pre-loading LSTM Predictor model...")
            Predictor.get_instance()
            print("[App Startup] Predictor ready for instant inference!")
        except Exception as e:
            print(f"[App Startup Warning] Could not pre-load model automatically: {e}")

    # --- Routes ---

    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('home'))
        return redirect(url_for('login'))

    @app.route('/healthz')
    def healthz():
        return jsonify({'status': 'ok'})

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if current_user.is_authenticated:
            return redirect(url_for('home'))

        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')

            # Validation
            if not username or not email or not password:
                flash('All fields are required.', 'danger')
                return render_template('signup.html', username=username, email=email)

            if len(password) < 6:
                flash('Password must be at least 6 characters long.', 'danger')
                return render_template('signup.html', username=username, email=email)

            if password != confirm_password:
                flash('Passwords do not match.', 'danger')
                return render_template('signup.html', username=username, email=email)

            if User.query.filter_by(email=email).first():
                flash('Email address is already registered.', 'danger')
                return render_template('signup.html', username=username, email=email)

            if User.query.filter_by(username=username).first():
                flash('Username is already taken.', 'danger')
                return render_template('signup.html', username=username, email=email)

            # Create user
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))

        return render_template('signup.html')

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for('home'))

        if request.method == 'POST':
            login_field = request.form.get('login_field', '').strip()
            password = request.form.get('password', '')

            if not login_field or not password:
                flash('Please provide your username/email and password.', 'danger')
                return render_template('login.html', login_field=login_field)

            # Allow login by username or email
            user = User.query.filter((User.username == login_field) | (User.email == login_field.lower())).first()

            if not user or not user.check_password(password):
                flash('Invalid credentials. Please check your username/email and password.', 'danger')
                return render_template('login.html', login_field=login_field)

            login_user(user)
            flash(f'Welcome back, {user.username}!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))

        return render_template('login.html')

    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out.', 'info')
        return redirect(url_for('login'))

    @app.route('/home')
    @login_required
    def home():
        # Get recent predictions for dashboard widgets
        recent_checks = PredictionHistory.query.filter_by(user_id=current_user.id)\
            .order_by(PredictionHistory.timestamp.desc()).limit(5).all()
            
        total_checks = PredictionHistory.query.filter_by(user_id=current_user.id).count()
        fake_count = PredictionHistory.query.filter_by(user_id=current_user.id, label='Fake').count()
        real_count = PredictionHistory.query.filter_by(user_id=current_user.id, label='Real').count()

        return render_template('home.html', 
                               user=current_user, 
                               recent_checks=recent_checks,
                               total_checks=total_checks,
                               fake_count=fake_count,
                               real_count=real_count)

    @app.route('/about')
    def about():
        return render_template('about.html')

    @app.route('/features')
    def features():
        return render_template('features.html')

    @app.route('/predict', methods=['POST'])
    @login_required
    def predict():
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '').strip()
        else:
            text = request.form.get('text', '').strip()

        if not text:
            return jsonify({'success': False, 'error': 'News article or headline text cannot be empty.'}), 400

        try:
            label, confidence, probability, token_count = predict_news(text)
            
            # Save to user prediction history
            history_record = PredictionHistory(
                user_id=current_user.id,
                input_text=text,
                label=label,
                confidence=confidence,
                probability=probability
            )
            db.session.add(history_record)
            db.session.commit()

            return jsonify({
                'success': True,
                'id': history_record.id,
                'label': label,
                'confidence': round(confidence * 100, 1),
                'probability': probability,
                'token_count': token_count,
                'timestamp': history_record.timestamp.strftime("%b %d, %Y %I:%M %p"),
                'snippet': (text[:120] + '...') if len(text) > 120 else text
            })
        except Exception as e:
            return jsonify({'success': False, 'error': f'Inference engine error: {str(e)}'}), 500

    @app.route('/history')
    @login_required
    def history():
        predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
            .order_by(PredictionHistory.timestamp.desc()).all()
            
        total_count = len(predictions)
        fake_count = sum(1 for p in predictions if p.label == 'Fake')
        real_count = sum(1 for p in predictions if p.label == 'Real')
        avg_confidence = round(sum(p.confidence for p in predictions) / total_count * 100, 1) if total_count > 0 else 0

        return render_template('history.html', 
                               predictions=predictions,
                               total_count=total_count,
                               fake_count=fake_count,
                               real_count=real_count,
                               avg_confidence=avg_confidence)

    @app.route('/history/delete/<int:item_id>', methods=['POST', 'DELETE'])
    @login_required
    def delete_history_item(item_id):
        record = PredictionHistory.query.filter_by(id=item_id, user_id=current_user.id).first()
        if not record:
            return jsonify({'success': False, 'error': 'Record not found'}), 404

        db.session.delete(record)
        db.session.commit()
        return jsonify({'success': True, 'id': item_id})

    @app.route('/history/clear', methods=['POST'])
    @login_required
    def clear_history():
        PredictionHistory.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        flash('All prediction history cleared.', 'info')
        return redirect(url_for('history'))

    @app.route('/api/metrics')
    def get_metrics():
        metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        return jsonify({'error': 'Metrics file not generated yet.'}), 404

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
