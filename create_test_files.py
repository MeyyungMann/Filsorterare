# Create a test script to generate sample files
import os
from pathlib import Path

def create_test_files():
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create files with different content types
    files_content = {
        # Code files
        "python_script.py": """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def main():
    for i in range(10):
        print(f"Fibonacci({i}) = {calculate_fibonacci(i)}")

if __name__ == "__main__":
    main()
""",
        
        # Meeting notes
        "team_meeting.txt": """
Team Meeting Notes - Project Kickoff
Date: 2024-03-15
Attendees: John, Sarah, Mike, Lisa

Agenda:
1. Project timeline review
2. Resource allocation
3. Risk assessment
4. Next steps

Key Decisions:
- Development phase: 3 months
- Weekly progress meetings
- Use Agile methodology
""",
        
        # Shopping list
        "grocery_list.txt": """
Weekly Grocery Shopping List

Produce:
- Apples (2kg)
- Bananas (1 bunch)
- Carrots (500g)
- Spinach (200g)

Dairy:
- Milk (2L)
- Eggs (12)
- Cheese (250g)

Bakery:
- Bread (1 loaf)
- Croissants (6)
""",
        
        # Todo list
        "project_tasks.txt": """
Project Tasks - Q2 2024

High Priority:
1. Complete database migration
2. Update API documentation
3. Fix critical bugs in payment system

Medium Priority:
1. Implement new features
2. Write unit tests
3. Code review

Low Priority:
1. Update README
2. Clean up old code
""",
        
        # Technical documentation
        "api_docs.txt": """
API Documentation v2.0

Authentication:
- Use JWT tokens
- Token expires in 24 hours
- Include in Authorization header

Endpoints:
- GET /api/users
- POST /api/users
- PUT /api/users/{id}
- DELETE /api/users/{id}

Response Format:
- JSON
- Status codes
- Error messages
""",
        
        # Personal notes
        "travel_plans.txt": """
Summer Vacation Plans 2024

Destination: Barcelona
Dates: July 15-22

Activities:
- Visit Sagrada Familia
- Beach day at Barceloneta
- Gothic Quarter tour
- Park Güell visit

Accommodation:
- Hotel booked
- Airport transfer arranged
""",
        
        # Work report
        "quarterly_report.txt": """
Q1 2024 Performance Report

Financial Highlights:
- Revenue: $1.2M
- Growth: 15% YoY
- Profit margin: 25%

Key Achievements:
- Launched new product
- Expanded to 3 new markets
- Hired 5 new team members

Challenges:
- Supply chain delays
- Market competition
""",
        
        # Italian Recipe
        "pasta_recipe.txt": """
Homemade Pasta Recipe

Ingredients:
- 2 cups flour
- 3 eggs
- 1 tbsp olive oil
- Salt to taste

Instructions:
1. Mix flour and salt
2. Add eggs and oil
3. Knead for 10 minutes
4. Rest for 30 minutes
5. Roll and cut
""",

        # Japanese Recipe
        "sushi_recipe.txt": """
Homemade Sushi Recipe

Ingredients:
- 2 cups sushi rice
- 1/4 cup rice vinegar
- Nori sheets
- Fresh salmon/tuna
- Cucumber
- Avocado
- Wasabi
- Soy sauce

Instructions:
1. Cook rice with vinegar
2. Prepare fish and vegetables
3. Place nori on bamboo mat
4. Spread rice evenly
5. Add fillings and roll
6. Slice and serve
""",

        # Indian Recipe
        "curry_recipe.txt": """
Butter Chicken Recipe

Ingredients:
- 500g chicken
- 2 tbsp butter
- 1 onion, diced
- 2 tomatoes
- 1 cup cream
- Garam masala
- Turmeric
- Cumin
- Coriander

Instructions:
1. Marinate chicken
2. Cook onions and spices
3. Add tomatoes and chicken
4. Simmer with cream
5. Garnish with coriander
""",

        # Thai Recipe
        "pad_thai_recipe.txt": """
Pad Thai Recipe

Ingredients:
- 200g rice noodles
- 2 eggs
- 100g tofu
- Bean sprouts
- Peanuts
- Tamarind paste
- Fish sauce
- Palm sugar
- Lime

Instructions:
1. Soak noodles
2. Prepare sauce
3. Stir-fry ingredients
4. Add noodles and sauce
5. Garnish with peanuts
""",
        
        # Study notes
        "math_notes.txt": """
Calculus Notes - Chapter 3

Derivatives:
- Power rule
- Chain rule
- Product rule
- Quotient rule

Applications:
- Optimization
- Related rates
- Curve sketching

Key Formulas:
- f'(x) = lim(h->0) [f(x+h)-f(x)]/h
""",
        
        # Fitness plan
        "workout_plan.txt": """
Weekly Workout Plan

Monday - Upper Body:
- Bench press: 3x10
- Pull-ups: 3x8
- Shoulder press: 3x12
- Bicep curls: 3x15

Wednesday - Lower Body:
- Squats: 3x12
- Deadlifts: 3x8
- Lunges: 3x10
- Calf raises: 3x20

Friday - Full Body:
- Burpees: 3x15
- Mountain climbers: 3x30s
- Plank: 3x60s
- Jump rope: 3x2min
""",

        # Additional recipe files
        "chicken_curry_recipe.txt": """
Chicken Curry Recipe

Ingredients:
- 600g chicken thighs
- 2 onions, finely chopped
- 4 garlic cloves
- 2 inch ginger
- 2 tomatoes
- 1 cup coconut milk
- Curry powder
- Cumin seeds
- Coriander leaves

Instructions:
1. Marinate chicken with spices
2. Sauté onions and garlic
3. Add chicken and cook
4. Simmer in coconut milk
5. Garnish with coriander
""",

        "vegetable_pasta_recipe.txt": """
Vegetable Pasta Recipe

Ingredients:
- 300g pasta
- 2 bell peppers
- 1 zucchini
- 1 eggplant
- Cherry tomatoes
- Garlic
- Olive oil
- Basil
- Parmesan

Instructions:
1. Cook pasta
2. Roast vegetables
3. Sauté garlic
4. Combine all ingredients
5. Top with cheese
""",

        # Additional documentation
        "setup_guide.txt": """
Project Setup Guide

Prerequisites:
- Python 3.8+
- Node.js 14+
- PostgreSQL 12+

Installation Steps:
1. Clone repository
2. Install dependencies
3. Configure database
4. Set environment variables
5. Run migrations

Development Setup:
- Install development tools
- Configure IDE
- Set up testing environment
""",

        "api_reference.txt": """
API Reference Guide

Authentication:
- OAuth 2.0
- API keys
- Rate limiting

Endpoints:
- Users API
- Products API
- Orders API
- Analytics API

Error Codes:
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
""",

        # Additional meeting notes
        "sprint_planning.txt": """
Sprint Planning Meeting

Date: 2024-03-20
Team: Development

Sprint Goals:
1. Complete user authentication
2. Implement payment system
3. Fix critical bugs

Tasks Assigned:
- John: Auth implementation
- Sarah: Payment integration
- Mike: Bug fixes
- Lisa: Testing
""",

        "retrospective.txt": """
Sprint Retrospective

What Went Well:
- Team collaboration
- Code quality
- Meeting deadlines

Areas for Improvement:
- Documentation
- Test coverage
- Communication

Action Items:
1. Update documentation
2. Add more unit tests
3. Schedule regular sync-ups
""",

        # Additional project tasks
        "bug_tasks.txt": """
Bug Fix Tasks

Critical:
1. Payment processing error
2. User data corruption
3. Security vulnerability

High Priority:
1. UI responsiveness
2. Data validation
3. Error handling

Medium Priority:
1. Performance optimization
2. Code cleanup
3. Documentation updates
""",

        # Additional study notes
        "physics_notes.txt": """
Physics Notes - Mechanics

Newton's Laws:
1. First Law: Inertia
2. Second Law: F=ma
3. Third Law: Action-Reaction

Key Concepts:
- Force
- Mass
- Acceleration
- Momentum
- Energy

Formulas:
- F = ma
- p = mv
- E = mc²
""",

        # Additional travel plans
        "japan_trip.txt": """
Japan Travel Plans

Destination: Tokyo
Dates: April 10-17

Itinerary:
- Day 1: Arrival, Shibuya
- Day 2: Tokyo Skytree
- Day 3: Tsukiji Market
- Day 4: Mount Fuji
- Day 5: Akihabara
- Day 6: Disneyland
- Day 7: Shopping

Accommodation:
- Hotel in Shinjuku
- Ryokan in Hakone
""",

        # Additional workout plans
        "yoga_routine.txt": """
Daily Yoga Routine

Morning:
- Sun salutations
- Standing poses
- Balance poses

Evening:
- Forward bends
- Hip openers
- Restorative poses

Breathing:
- Pranayama
- Meditation
- Relaxation
""",

        # Additional reports
        "monthly_report.txt": """
March 2024 Monthly Report

Sales Performance:
- Total revenue: $450K
- New customers: 150
- Repeat customers: 75%

Marketing:
- Campaign results
- Social media growth
- Customer feedback

Operations:
- Efficiency metrics
- Cost analysis
- Team performance
""",

        # Additional medical files
        "patient_notes.txt": """
Patient Medical Notes

Patient ID: P12345
Date: 2024-03-15

Chief Complaint:
- Persistent cough for 2 weeks
- Shortness of breath
- Low-grade fever

Vital Signs:
- BP: 120/80
- HR: 88
- Temp: 37.2°C
- SpO2: 96%

Assessment:
- Probable viral upper respiratory infection
- Mild bronchospasm
- No signs of pneumonia

Plan:
1. Rest and hydration
2. OTC cough suppressant
3. Follow-up in 1 week if symptoms persist
""",

        "lab_results.txt": """
Laboratory Test Results

Patient: John Smith
Date: 2024-03-15

Complete Blood Count:
- WBC: 7.5 K/µL
- RBC: 4.8 M/µL
- HGB: 14.2 g/dL
- PLT: 250 K/µL

Chemistry Panel:
- Glucose: 95 mg/dL
- BUN: 15 mg/dL
- Creatinine: 0.9 mg/dL
- Na: 140 mEq/L
- K: 4.0 mEq/L

Lipid Panel:
- Total Cholesterol: 180 mg/dL
- HDL: 45 mg/dL
- LDL: 100 mg/dL
- Triglycerides: 150 mg/dL
""",

        "treatment_plan.txt": """
Treatment Plan for Diabetes Management

Patient: Sarah Johnson
Diagnosis: Type 2 Diabetes
Date: 2024-03-15

Current Medications:
1. Metformin 1000mg BID
2. Glipizide 5mg QD
3. Lisinopril 10mg QD

Lifestyle Modifications:
- Daily exercise: 30 minutes
- Carbohydrate counting
- Regular blood glucose monitoring

Follow-up Schedule:
- Monthly clinic visits
- Quarterly HbA1c testing
- Annual comprehensive exam

Goals:
1. HbA1c < 7.0%
2. Fasting glucose 80-130 mg/dL
3. Postprandial glucose < 180 mg/dL
""",

        # Additional coding files
        "data_processor.py": """
import pandas as pd
import numpy as np
from typing import List, Dict

class DataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        \"\"\"Load data from CSV file.\"\"\"
        self.data = pd.read_csv(self.data_path)
        return self.data
        
    def clean_data(self) -> pd.DataFrame:
        \"\"\"Clean and preprocess the data.\"\"\"
        if self.data is None:
            raise ValueError("Data not loaded")
            
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())
        
        return self.data
        
    def transform_data(self, columns: List[str]) -> pd.DataFrame:
        \"\"\"Apply transformations to specified columns.\"\"\"
        if self.data is None:
            raise ValueError("Data not loaded")
            
        for col in columns:
            if col in self.data.columns:
                self.data[col] = np.log1p(self.data[col])
                
        return self.data
""",

        "api_client.py": """
import requests
from typing import Dict, Any, Optional
import json

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        \"\"\"Make GET request to API endpoint.\"\"\"
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
        
    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Make POST request to API endpoint.\"\"\"
        url = f"{self.base_url}/{endpoint}"
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
        
    def put(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Make PUT request to API endpoint.\"\"\"
        url = f"{self.base_url}/{endpoint}"
        response = self.session.put(url, json=data)
        response.raise_for_status()
        return response.json()
""",

        "database_schema.sql": """
-- Database schema for medical records system

CREATE TABLE patients (
    patient_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    date_of_birth DATE NOT NULL,
    gender VARCHAR(10),
    contact_number VARCHAR(20),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE medical_records (
    record_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    visit_date DATE NOT NULL,
    diagnosis TEXT,
    treatment TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE prescriptions (
    prescription_id SERIAL PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    medication_name VARCHAR(100) NOT NULL,
    dosage VARCHAR(50),
    frequency VARCHAR(50),
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patient_name ON patients(last_name, first_name);
CREATE INDEX idx_visit_date ON medical_records(visit_date);
""",

        "test_cases.py": """
import unittest
from data_processor import DataProcessor
from api_client import APIClient

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = DataProcessor("test_data.csv")
        
    def test_load_data(self):
        data = self.processor.load_data()
        self.assertIsNotNone(data)
        self.assertTrue(len(data) > 0)
        
    def test_clean_data(self):
        self.processor.load_data()
        cleaned_data = self.processor.clean_data()
        self.assertFalse(cleaned_data.duplicated().any())
        self.assertFalse(cleaned_data.isnull().any().any())
        
class TestAPIClient(unittest.TestCase):
    def setUp(self):
        self.client = APIClient("https://api.example.com", "test_key")
        
    def test_get_request(self):
        response = self.client.get("users", {"id": 1})
        self.assertIn("id", response)
        self.assertIn("name", response)
        
    def test_post_request(self):
        data = {"name": "Test User", "email": "test@example.com"}
        response = self.client.post("users", data)
        self.assertEqual(response["name"], data["name"])
""",

        "config.yaml": """
# Application Configuration

database:
  host: localhost
  port: 5432
  name: medical_records
  user: admin
  password: ${DB_PASSWORD}

api:
  base_url: https://api.medical.example.com
  version: v1
  timeout: 30
  retry_attempts: 3

logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/app.log

security:
  jwt_secret: ${JWT_SECRET}
  token_expiry: 3600
  allowed_origins:
    - https://medical.example.com
    - https://admin.medical.example.com
"""
    }
    
    # Create the files
    for filename, content in files_content.items():
        file_path = test_dir / filename
        # Use UTF-8 encoding when writing files
        file_path.write_text(content.strip(), encoding='utf-8')
        print(f"Created: {file_path}")

if __name__ == "__main__":
    create_test_files()