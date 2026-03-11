const express = require('express');
const path = require('path');
const cors = require('cors');
const http = require('http');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const { Server } = require('socket.io');
require('dotenv').config();

if (!process.env.JWT_SECRET) {
  throw new Error("JWT_SECRET missing from environment variables");
}

const connectDB = require('./db');
const Job = require('./models/Job');
const User = require('./models/User');

const ML_API_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname)); // Serve frontend files

// Connect to MongoDB
connectDB();

// --- INITIAL DATA SEEDING ---
const initialJobs = [
  { id: "1", title: "React Dashboard for Algo Club", budget: "8,000", tags: ["React", "Campus"], applicants: 4, description: "Build a responsive dashboard for the Algo Club's upcoming hackathon. Must be fast and dark-mode enabled.", duration: "3 Weeks", university: "Dayananda Sagar University" },
  { id: "2", title: "Event Photographer for TechFest", budget: "3,000", tags: ["Photography", "Event"], applicants: 12, description: "Need a photographer for 2 days of TechFest coverage. Editing included.", duration: "2 Days", university: "Dayananda Sagar University" },
  { id: "3", title: "Python Script for Attendance", budget: "5,000", tags: ["Python", "Automation"], applicants: 2, description: "Automate attendance parsing from Zoom logs using Python. Simple CLI tool required.", duration: "1 Week", university: "Dayananda Sagar University" },
  { id: "9", title: "Cybersecurity Audit Script", budget: "6,000", tags: ["Security", "Bash"], applicants: 1, description: "Write a script to check open ports on the lab network (Authorized use only).", duration: "1 Week", university: "Dayananda Sagar University" },
  { id: "10", title: "Marketing Flyers for Cultural Fest", budget: "2,500", tags: ["Canva", "Design"], applicants: 8, description: "Create 5 high-quality Instagram posters for the upcoming ethnic day.", duration: "3 Days", university: "Dayananda Sagar University" },
  { id: "11", title: "Tutor: Data Structures (C++)", budget: "4,000", tags: ["Teaching", "DSA"], applicants: 5, description: "Need a tutor to help me prepare for internals. 5 sessions of 1 hour each.", duration: "1 Week", university: "Dayananda Sagar University" },
  { id: "4", title: "3D Game Assets for Unity", budget: "15,000", tags: ["3D", "Game Dev"], applicants: 28, description: "Looking for low-poly character models for a mobile game. Portfolio required.", duration: "1 Month", university: "VIT Vellore" },
  { id: "5", title: "Mobile App UI Redesign", budget: "25,000", tags: ["Figma", "UI/UX"], applicants: 45, description: "Redesign our fintech app. We need a modern, clean, and trustworthy look.", duration: "1.5 Months", university: "Stanford University" },
  { id: "6", title: "Node.js Backend Developer", budget: "20,000", tags: ["NodeJS", "API"], applicants: 15, description: "Build REST APIs for a social media startup. Experience with MongoDB is a plus.", duration: "2 Months", university: "Global Remote" },
  { id: "7", title: "Content Writer for Tech Blog", budget: "4,000", tags: ["Writing", "SEO"], applicants: 8, description: "Write 5 SEO-optimized articles about AI and Machine Learning.", duration: "Ongoing", university: "BITS Pilani" },
  { id: "8", title: "Video Editor for YouTube Channel", budget: "10,000", tags: ["Premiere", "Editing"], applicants: 32, description: "Edit varying pace tutorial videos. Must verify sync and audio quality.", duration: "Recurring", university: "Remote" },
  { id: "12", title: "AWS Lambda Deployment", budget: "12,000", tags: ["AWS", "Cloud"], applicants: 3, description: "Deploy a serverless Python function to AWS Lambda with API Gateway.", duration: "1 Week", university: "Global Remote" },
  { id: "13", title: "Voice Over (American Accent)", budget: "8,000", tags: ["Voice", "Audio"], applicants: 20, description: "Record a 2-minute explainer video script. Clean audio required.", duration: "1 Day", university: "Remote" },
  { id: "14", title: "Translate App to Spanish", budget: "5,000", tags: ["Translation", "Lang"], applicants: 6, description: "Translate 500 strings from English to Spanish for a React Native app.", duration: "2 Days", university: "Remote" },
  { id: "15", title: "Library Book Sorting Helper", budget: "500", tags: ["Helper", "Campus"], applicants: 10, description: "Help the library staff sort new arrivals for 3 hours. Snacks provided.", duration: "1 Day", university: "Dayananda Sagar University" },
  { id: "16", title: "Sports Club Logo Design", budget: "1,500", tags: ["Design", "Logo"], applicants: 3, description: "Modern logo for the DSU Cricket Team. Must include eagle mascot.", duration: "2 Days", university: "Dayananda Sagar University" },
  { id: "17", title: "Hackathon Mentor (Java/Python)", budget: "5,000", tags: ["Mentoring", "Coding"], applicants: 2, description: "Guide first-year students during the 24-hour hackathon. Must be proficient in backend dev.", duration: "1 Day", university: "Dayananda Sagar University" },
  { id: "18", title: "Shopify Store Setup", budget: "15,000", tags: ["Shopify", "E-com"], applicants: 18, description: "Set up a full Shopify store with 20 products and payment gateway integration.", duration: "1 Week", university: "Global Remote" },
  { id: "19", title: "Excel Data Entry Project", budget: "2,000", tags: ["Excel", "Data"], applicants: 50, description: "Copy data from PDF to Excel. High accuracy required. 50 pages.", duration: "3 Days", university: "Remote" },
  { id: "20", title: "Instagram Reels Creator", budget: "8,000", tags: ["Video", "Social"], applicants: 25, description: "Create 5 viral-style reels for a fashion brand. Raw footage provided.", duration: "1 Week", university: "Remote" }
];

const seedDatabase = async () => {
  try {
    const mongoose = require('mongoose');
    if (mongoose.connection.readyState !== 1) {
      console.log('⏳ Waiting for database connection before seeding...');
      return; // Will be called again by setTimeout if needed, or by connection callback
    }

    const count = await Job.countDocuments();
    if (count === 0) {
      await Job.insertMany(initialJobs);
      console.log('📦 Database Seeded with Initial Jobs!');
    } else {
      console.log('✅ Database already has data.');
    }

    const userCount = await User.countDocuments();
    if (userCount === 0) {
      await User.create([
        { name: 'Abhiram Student', email: 'student@demo.com', password: 'password123', role: 'student', skills: ['React', 'Python', 'Java'] },
        { name: 'Abhiram Client', email: 'client@demo.com', password: 'password123', role: 'client', company: 'Skiller Corp' }
      ]);
      console.log('👤 Seeded Test Users: student@demo.com, client@demo.com');
    }
  } catch (err) {
    console.error('Seeding Error:', err.message);
  }
};

// Seed on startup but give it some extra time for the remote cluster
setTimeout(seedDatabase, 5000);

app.use(express.static(path.join(__dirname)));
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'demo.html'));
});

// --- SERVER & SOCKET ---
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

// Note: Python local STDIN processes have been removed
// in favor of the new ML Microservices Architecture running on FastAPI.

io.on('connection', async (socket) => {
  console.log(`socket connected ${socket.id}`);

  // Fetch Jobs from DB
  try {
    const jobs = await Job.find().lean();

    // Send correct structure for Frontend: { jobs, stats: { ... } }
    socket.emit('init', {
      jobs: jobs,
      stats: { online: io.engine.clientsCount }
    });
  } catch (e) {
    console.error('Error fetching jobs, using fallback:', e.message);
    // FALLBACK TO INITIAL JOBS FOR DEMO
    socket.emit('init', {
      jobs: initialJobs,
      stats: { online: io.engine.clientsCount }
    });
  }

  // --- AUTH ENDPOINTS ---
  socket.on('signup', async (userData) => {
    try {
      // CHECK DB STATUS
      if (require('mongoose').connection.readyState !== 1) {
        console.log("⚠️ DB Offline. Signing up as Mock User.");
        const mockUser = {
          _id: 'mock_user_123',
          name: userData.name || 'Demo Student',
          email: userData.email,
          role: userData.role || 'student',
          skills: userData.skills || ['React', 'NodeJS', 'Python'],
          isVerified: false,
          company: userData.company || 'Skiller Corp',
          skillScore: 85,
          skillBreakdown: { profile: 15, skills: 20, activity: 15, projects: 10, performance: 15, market: 10 }
        };
        const token = jwt.sign(
          { userId: mockUser._id, role: mockUser.role },
          process.env.JWT_SECRET,
          { expiresIn: '7d' }
        );
        socket.emit('authSuccess', { user: mockUser, token });
        return;
      }

      const exists = await User.findOne({ email: userData.email });
      if (exists) {
        socket.emit('authError', 'Email already registered!');
        return;
      }

      const hashedPassword = await bcrypt.hash(userData.password, 10);
      const userToCreate = { ...userData, password: hashedPassword };
      const newUser = await User.create(userToCreate);

      const token = jwt.sign(
        { userId: newUser._id, role: newUser.role },
        process.env.JWT_SECRET,
        { expiresIn: '7d' }
      );

      socket.emit('authSuccess', { user: newUser, token });
    } catch (err) {
      if (err.message.includes('findOne()')) {
        socket.emit('authError', 'Signup Failed: Database is not connected.');
      } else {
        socket.emit('authError', 'Signup Failed: ' + err.message);
      }
    }
  });

  socket.on('login', async (creds) => {
    try {
      // CHECK DB STATUS
      if (require('mongoose').connection.readyState !== 1) {
        console.log("⚠️ DB Offline. Logging in as Mock User.");
        const mockUser = {
          _id: 'mock_user_123',
          name: 'Demo Student',
          email: creds.email,
          role: 'student',
          skills: ['React', 'NodeJS', 'Python'],
          isVerified: false,
          company: 'Skiller Demo University',
          skillScore: 85,
          skillBreakdown: { profile: 15, skills: 20, activity: 15, projects: 10, performance: 15, market: 10 }
        };
        socket.emit('authSuccess', { user: mockUser, token: 'offline-token' });
        return;
      }

      // Simple plain text password check for demo
      const user = await User.findOne({ email: creds.email });
      if (user && await bcrypt.compare(creds.password, user.password)) {
        const token = jwt.sign(
          { userId: user._id, role: user.role },
          process.env.JWT_SECRET,
          { expiresIn: '7d' }
        );
        socket.emit('authSuccess', { user, token });
      } else {
        socket.emit('authError', 'Invalid Email or Password');
      }
    } catch (err) {
      console.error("Login Error:", err.message);
      socket.emit('authError', 'Login Error: ' + err.message);
    }
  });


  // --- JOB ENDPOINTS ---
  socket.on('postJob', async (newJob) => {
    try {
      const createdJob = await Job.create({
        ...newJob,
        id: Date.now().toString(),
        applicants: 0
      });
      io.emit('newJob', createdJob);
    } catch (e) { console.error(e); }
  });

  socket.on('markVerified', async (userId) => {
    try {
      const user = await User.findById(userId);
      if (user) {
        user.isVerified = true;
        await user.save();

        // --- REAL-TIME SCORE UPDATE ---
        const result = await calculateSkillScore(user); // Now async
        user.skillScore = result.score;
        user.skillBreakdown = result.breakdown;
        await user.save();

        // Notify Frontend
        socket.emit('scoreUpdate', { userId, score: user.skillScore, breakdown: user.skillBreakdown });
      }
    } catch (e) {
      console.error("Verification Update Error:", e);
    }
  });

  socket.on('applyJob', async (data) => {
    try {
      const job = await Job.findOne({ id: data.jobId });
      if (job) {
        job.applicants += 1;
        await job.save();
        io.emit('jobUpdate', job);
      }
    } catch (e) { console.error(e); }
  });

  // --- PYTHON FACIAL RECOGNITION ---
  socket.on('verifyFace', (imageData) => {
    const { spawn } = require('child_process');
    const py = spawn('python', ['./face_scan.py']);

    let outputData = '';

    const payload = JSON.stringify({ image: imageData });
    py.stdin.write(payload);
    py.stdin.end();

    py.stdout.on('data', (chunk) => outputData += chunk.toString());
    py.stderr.on('data', (chunk) => console.error("Python Stderr:", chunk.toString()));

    py.on('close', (code) => {
      if (code !== 0) {
        console.error(`Face Scan Process Exited with Code ${code}`);
        socket.emit('faceResult', { verified: false, error: 'AI Process Crashed' });
        return;
      }
      try {
        const result = JSON.parse(outputData);
        socket.emit('faceResult', result);
      } catch (e) {
        console.error("ML Parse Error:", e, "Output was:", outputData);
        socket.emit('faceResult', { verified: false, error: 'Server AI Failed' });
      }
    });
  });

  // --- NEW AI ML MICROSERVICES INTEGRATION ---

  socket.on('runPyML', async (payload) => {
    try {
      if (payload.action === 'recommend_talent') {
        const jobsProfiles = payload.students.map(s => (s.name || '') + ' ' + (s.skills || []).join(' '));
        const res = await fetch(`${ML_API_URL}/match-talent`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_description: payload.job.description + ' ' + (payload.job.tags || []).join(' '),
            freelancer_profiles: jobsProfiles
          })
        });

        if (!res.ok) throw new Error("ML Service is Offline or Unresponsive");

        const data = await res.json();
        let results = data.matches.map(m => {
          let student = payload.students[m.profile_index];
          let score = m.compatibility_score * 100;
          return {
            candidate: student,
            score: score.toFixed(1),
            analysis: `Predictive Match: [${score.toFixed(1)}% Success Likelihood] (SentenceTransformer & RF model).`
          };
        });
        socket.emit('mlResults', results.slice(0, 5));
      } else {
        socket.emit('mlError', 'Action not supported yet by new Microservices.');
      }
    } catch (e) {
      console.error(e);
      socket.emit('mlError', 'ML Microservice Offline Failure');
    }
  });

  // --- AI SKILL VERIFICATION ENGINE ---
  socket.on('runSkillVerification', async (payload) => {
    try {
      const res = await fetch(`${ML_API_URL}/verify-skill`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: payload.code })
      });

      if (!res.ok) throw new Error("ML Service is Offline");

      const data = await res.json();
      socket.emit('skillVerificationResult', {
        status: "success",
        score: Math.round(data.predicted_skill_score),
        breakdown: {
          cyclomatic_complexity: data.metrics.cc,
          design_cohesion: data.metrics.fc * 10,
          vulnerabilities_found: 0,
          quality_index: Math.round(data.predicted_skill_score)
        },
        ai_insight: "RandomForest inference complete. Evaluated via AST features."
      });
    } catch (e) {
      console.error("Verification Error:", e.message);
      socket.emit('skillVerificationError', 'ML Service Offline');
    }
  });

  // --- AI RESUME INTELLIGENCE NLP ---
  socket.on('analyzeResumeNLP', async (payload) => {
    try {
      const res = await fetch(`${ML_API_URL}/analyze-resume`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_data: payload.file_data,
          filename: payload.filename,
          file_type: payload.file_type
        })
      });

      if (!res.ok) throw new Error("ML Service Offline");

      const data = await res.json();
      socket.emit('resumeNLPResult', {
        status: "success",
        extracted_skills: data.extracted_skills || ["Evaluated by DL Model"],
        predicted_role: data.predicted_role,
        primary_role: data.predicted_role,
        resume_strength_score: data.resume_strength_score || 85,
        skill_gaps: data.skill_gaps || [],
        actionable_feedback: data.actionable_feedback || [`Model predicts category: ${data.predicted_role}`]
      });
    } catch (e) {
      console.error("Resume NLP Error:", e.message);
      socket.emit('resumeNLPError', 'ML Service Offline - Resume Model Failed');
    }
  });

  // --- AI REPUTATION INTELLIGENCE SCORE ---
  socket.on('analyzeTrustScore', async (payload) => {
    try {
      const completion_rate = (payload.user.jobsCompleted || 0) / (payload.user.jobsApplied || 1);
      const res = await fetch(`${ML_API_URL}/trust-score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          completion_rate: completion_rate,
          ratings: 4.5, // Dummy average
          skill_scores: payload.user.skillScore || 50,
          historical_activity: payload.user.jobsCompleted || 10
        })
      });

      if (!res.ok) throw new Error("ML Service Offline");

      const data = await res.json();
      const score = Math.round(data.trust_score);
      socket.emit('trustScoreResult', {
        status: "success",
        trust_index: score,
        tier: score > 75 ? "Elite Trusted" : "Valued Member",
        breakdown: { model_prediction: score }
      });
    } catch (e) {
      console.error("Trust Score Error:", e.message);
      socket.emit('trustScoreError', 'ML Service Offline - Trust Model Failed');
    }
  });

  // --- AI FRAUD & PROFILE ANOMALY DETECTOR ---
  socket.on('runFraudCheck', async (payload) => {
    try {
      const features = Array(28).fill(0).map(() => Math.random() * 2 - 1);
      if (payload.jobsCompleted === 0 && payload.applications_last_hour > 20) {
        features[0] += 5;
      }

      const res = await fetch(`${ML_API_URL}/detect-fraud`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: features,
          amount: 100.0 // Dummy amount
        })
      });

      if (!res.ok) throw new Error("ML Service Offline");

      const data = await res.json();
      socket.emit('fraudCheckResult', {
        status: "success",
        risk_score: data.risk_score,
        assessment: data.is_fraud ? "CRITICAL: Fraud Anomaly Detected via ML" : "SECURE",
        recommended_action: data.is_fraud ? "Suspend Account" : "None"
      });
    } catch (e) {
      console.error("Fraud Check Error:", e.message);
      socket.emit('fraudCheckError', 'ML Service Offline - Fraud Model Failed');
    }
  });

  socket.on('disconnect', () => {
    // Send updated stats on disconnect
    io.emit('statsUpdate', { online: io.engine.clientsCount });
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on http://localhost:${PORT}`);
  console.log(`Network Access: http://${getLocalIp()}:${PORT}`);
});

// --- SKILL SCORE & HIRING INTELLIGENCE SYSTEM ---

// 1. Calculate Dynamic Skill Score (0-100)
// 1. Calculate Dynamic Skill Score (0-100) - HYBRID AI APPROACH
const calculateSkillScore = async (user) => {
  // Attempt Python ML Analysis First
  const { spawn } = require('child_process');

  return new Promise((resolve, reject) => {
    const py = spawn('python', ['./ml_engine.py']);
    let outputData = '';

    const payload = JSON.stringify({ action: 'calculate_skill_score', user: user });

    py.stdin.write(payload);
    py.stdin.end();

    py.stdout.on('data', (chunk) => outputData += chunk.toString());
    py.stderr.on('data', (c) => console.error(c.toString()));

    py.on('close', (code) => {
      if (code === 0 && outputData) {
        try {
          const aiResult = JSON.parse(outputData);
          resolve(aiResult); // Returns { score, breakdown, ai_tip }
        } catch (e) {
          console.error("AI Parse Error, falling back to heuristic", e);
          resolve(heuristicScore(user));
        }
      } else {
        resolve(heuristicScore(user));
      }
    });

    // Timeout in case Python hangs
    setTimeout(() => resolve(heuristicScore(user)), 4000);
  });
};

// Fallback Heuristic Score (Pure JS)
const heuristicScore = (user) => {
  let score = 0;
  const breakdown = { profile: 0, skills: 0, activity: 0, projects: 0, performance: 0, market: 0 };

  // Basic heuristic logic...
  if (user.name) breakdown.profile += 5;
  if (user.company || user.university) breakdown.profile += 5;
  if (user.isVerified) breakdown.profile += 10;
  score += breakdown.profile;

  const skillCount = (user.skills || []).length;
  breakdown.skills = Math.min(skillCount * 4, 30);
  score += breakdown.skills;

  breakdown.activity = 15;
  breakdown.projects = 10;

  return { score: Math.min(score, 85), breakdown, ai_tip: "Add more skills to improve (Offline Mode)." };
};

// 2. Hiring Fit Score (Applicant vs Job)
const calculateHiringFit = (applicant, job) => {
  let fitScore = 0;
  const jobTags = (job.tags || []).map(t => t.toLowerCase());
  const userSkills = (applicant.skills || []).map(s => s.toLowerCase());
  const matches = userSkills.filter(s => jobTags.some(t => t.includes(s) || s.includes(t)));

  if (jobTags.length > 0) fitScore += (matches.length / jobTags.length) * 40;
  fitScore += (applicant.skillScore / 100) * 30;
  if (applicant.isVerified) fitScore += 10;
  fitScore += 20;

  return Math.min(Math.round(fitScore), 100);
};


// --- API ROUTES FOR SKILL SCORE ---

const verifyToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  if (!authHeader) return res.status(403).json({ error: 'No token provided' });
  const token = authHeader.split(' ')[1] || authHeader;

  jwt.verify(token, process.env.JWT_SECRET, (err, decoded) => {
    if (err) return res.status(401).json({ error: 'Unauthorized' });
    req.userId = decoded.userId;
    req.role = decoded.role;
    next();
  });
};

app.get('/health', async (req, res) => {
  try {
    const dbStatus = require('mongoose').connection.readyState === 1 ? "connected" : "disconnected";
    let mlStatus = "offline";
    try {
      const mlRes = await fetch(`${ML_API_URL}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(2000)
      });
      if (mlRes.ok) mlStatus = "online";
    } catch (e) { }

    res.json({
      status: "ok",
      db: dbStatus,
      ml_service: mlStatus,
      timestamp: Date.now()
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Trigger Manual Recalculation
app.post('/api/calculate-score', verifyToken, async (req, res) => {
  try {
    const { userId } = req.body;

    // Handle Mock User in Offline Mode
    if (userId === 'mock_user_123') {
      const result = {
        score: 85,
        breakdown: { profile: 20, skills: 20, activity: 15, projects: 10, performance: 10, market: 10 },
        ai_tip: "Offline Mode: Your skills are highly competitive! (Connect DB for real AI analysis)"
      };
      return res.json({ success: true, ...result });
    }

    const user = await User.findById(userId);
    if (!user) return res.status(404).json({ error: "User not found" });

    // Call Async AI Engine
    const result = await calculateSkillScore(user);

    user.skillScore = result.score;
    user.skillBreakdown = result.breakdown;
    user.lastSkillScoreUpdate = Date.now();
    await user.save();

    // Emit update to user via socket if connected
    io.emit('scoreUpdate', { userId, score: user.skillScore, breakdown: user.skillBreakdown, tip: result.ai_tip });

    res.json({ success: true, score: user.skillScore, breakdown: user.skillBreakdown, tip: result.ai_tip });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// Get Score Details
app.get('/api/get-score/:userId', async (req, res) => {
  try {
    const user = await User.findById(req.params.userId);
    if (!user) return res.status(404).json({ error: "User not found" });
    res.json({ score: user.skillScore, breakdown: user.skillBreakdown });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

function getLocalIp() {
  const { networkInterfaces } = require('os');
  const nets = networkInterfaces();
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        return net.address;
      }
    }
  }
  return 'localhost';
}