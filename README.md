Netra – Real-Time Behavioural Safety Detection System

Netra is an AI-powered real-time behavioural safety system designed to convert CCTV feeds into proactive alerts for violence, panic, medical emergencies, child distress, and suspicious abandoned objects.

Developed for Hack for Social Cause (HSC) – VBYLD 2026,
Netra aims to strengthen Governance & Civic Technology and improve urban safety across Maharashtra and India.

⭐ Features
🔹 Violence Detection

Detects aggression, fights, pushing behaviour.

🔹 Crowd Panic / Stampede Risk

Identifies sudden chaotic movement patterns.

🔹 Fall / Collapse Detection

Useful for medical emergencies in public places.

🔹 Child Distress / Forced Movement Detection

Flags unsafe or forceful interactions.

🔹 Suspicious Abandoned Object Detection

Detects unattended bags or items in vulnerable areas.

🔹 Severity Scoring System

Ranks incidents as Low / Medium / High for fast response.

🔹 Real-Time Dashboard (Streamlit)

Shows live alerts, timestamps, and incident logs.

⭐ Tech Stack
| Component          | Technology               |
| ------------------ | ------------------------ |
| Object Detection   | **YOLOv5**               |
| CV Processing      | **OpenCV**               |
| Logic & Backend    | **Python 3.10+**         |
| Dashboard          | **Streamlit**            |
| Behaviour Analysis | Custom motion heuristics |


Component	Technology
Object Detection	YOLOv5
CV Processing	OpenCV
Logic & Backend	Python 3.10+
Dashboard	Streamlit
Behaviour Analysis	Custom motion heuristics


⭐ How It Works

CCTV feed or sample video is given to the system.

YOLOv5 performs object & person detection.

Motion and spatial patterns are analysed to understand behaviour.

Severity scoring engine ranks each detected event.

Real-time alerts appear on the Streamlit dashboard.

All incidents get logged with timestamps.


Installation
1. Clone the repository
git clone https://github.com/Ritz-V/Netra.git
cd Netra

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py


⭐ Ethical & Privacy Considerations

Netra is designed to prioritize ethics and privacy:

✔ No facial recognition
✔ No identity tracking
✔ No biometric storage
✔ Only behaviour & motion analysis
✔ Intended for public safety & smart governance
✔ Built for positive social impact aligned with UN SDGs 3, 11, and 16

⭐ Use Cases

Smart City Command Centres

Railway & Metro Stations

College & School Campuses

Bus Stands & High Footfall Zones

Malls & Markets

Parks & Public Spaces

Disaster & Emergency Management Systems

⭐ Author

Riddhi Vyas
Cusrow Wadia Institute of Technology, Pune
Hack for Social Cause – VBYLD 2026
GitHub: https://github.com/Ritz-V

⭐ License

This project is shared for educational & hackathon purposes.
Contact the author before any commercial or production use.
