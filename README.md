# ParkingMeter.co.uk

A **smart parking** system with real-time availability tracking, built as a final year Computer Science project at the University of Hull (2025/26).

The system uses edge-based computer vision to monitor carpark occupancy in real time, pushing live updates to a public-facing web platform where drivers can view availability and make reservations.

**Live site: [parkingmeter.co.uk](https://parkingmeter.co.uk)**

---

<img width="1716" height="1275" alt="parking meter help" src="https://github.com/user-attachments/assets/92af4372-c30b-4f16-8e67-9ab642e9bb61" />

---

## What it does

- Shows live carpark availability on an interactive map
- Detects vehicle occupancy using YOLO computer vision on a live camera feed
- Lets drivers reserve a parking spot in advance
- Gives carpark owners a staff dashboard to manage cameras, define parking spots, and view occupancy history
- Updates in real time via Supabase WebSocket subscriptions - no page refresh needed

---

## How to use

### Location permission

When you first open the site, your browser will ask for your location. **Please accept this** - it is only used to centre the map on your area and show nearby carparks. No location data is stored or sent anywhere beyond your local browser session.



---

### Log in with test credentials

You can explore the full site including the staff dashboard using the test account below. There is a log in button in the bottom left corner of the site but it should pop up if its your first time loading parkingmeter.co.uk.

<img height="350" alt="loginin" src="https://github.com/user-attachments/assets/547afbbb-3290-4184-9ca3-aefe475f7023" />



The staff+customer account gives access to the owner dashboard where you can see live occupancy, camera feeds, and the 30-day occupancy history chart.

---

### Making a reservation

1. Allow location or browse the map manually
2. Click a carpark to see live availability
3. Select a spot and choose a time
4. Complete the booking flow - payment is in test mode so no real money is charged

---

## Technical details

| | |
|---|---|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Python, Flask |
| **Database** | Supabase (PostgreSQL) |
| **Auth** | Supabase Auth (bcrypt password hashing) locked due to traffic and prevent billing issues|
| **Computer vision** | YOLO26s via Ultralytics - selected after benchmarking 45 models to find best fit |
| **Video capture** | OpenCV |
| **Payments** | Stripe (To Be Added)|
| **Map** | OpenStreetMap API |

### Computer vision

The detection pipeline captures frames from a **live camera feed**, runs YOLO inference every few seconds, and checks whether any detected vehicle bounding box falls inside a** user-defined parking spot** polygon using a point-in-polygon algorithm. Only occupancy state changes are pushed to the database to avoid unnecessary traffic.

 was chosen as the default model after a benchmarking framework tested 45 models across 8 architecture families against 500 frames of real carpark footage. It averaged 17.2 detections per frame against a ground truth of 16-17 vehicles, at 7.8 FPS on CPU with 0.69 average confidence.

### **yolo26s**  - Perfect
[![Good detection - yolo26s](https://img.youtube.com/vi/cD1-iss17WE/maxresdefault.jpg)](https://youtu.be/cD1-iss17WE)

### **Custom Model** - Genuinely Awfull 
[![Poor detection - custom model](https://img.youtube.com/vi/mSvLeP8Js74/maxresdefault.jpg)](https://youtu.be/mSvLeP8Js74)

### Hosting

The site runs on a local laptop behind a **Cloudflare Tunnel**, which provides free HTTPS, DDoS protection, and global CDN distribution with **zero infrastructure cost**. There is no traditional cloud server. The only ongoing costs are the **Supabase Pro plan** and the **domain**.
  
Over the first 30 days the site received **1,420 unique visitors** and **1.15 million total requests** from 58 countries, with 100% uptime.

---

## Project context

This is a third year Computer Science dissertation project at the **University of Hull**, submitted April 2026. The codebase covers a distributed detection server, a public-facing booking platform, a staff management dashboard, and a standalone benchmarking framework for evaluating computer vision models and misc.

The project is not intended for commercial use in its current state. Row Level Security on the database is disabled during the prototype phase and would need to be enabled before any real deployment.

Misc
The original domain I wanted was taken and overpriced, so parkingmeter.co.uk ended up being the better choice anyway - it sounds more official and is easier to remember. I genuinely think very little marketing would be needed to get drivers using it, the organic traffic backs that up. The harder problem is the supply side. Getting carpark owners to either install the detection server or share their existing camera feeds is the real barrier, and my local Council (Hull City Council) demonstrated exactly that - they wouldn't provide even a sample video clip let alone a live feed. If this project had any real backing behind it, most of that investment would go into partnerships with carpark operators rather than the technology itself. Offering free use of the platform in exchange for camera access would be a reasonable starting point.
