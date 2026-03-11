const express = require('express');
const { db, init } = require('../db');
const jwt = require('jsonwebtoken');
const SECRET = process.env.JWT_SECRET || 'dev-secret';

const router = express.Router();

(async () => { await init(); })();

async function auth(req, res, next) {
  const h = req.headers.authorization;
  if (!h) return res.status(401).json({ message: 'Missing token' });
  const token = h.split(' ')[1];
  try {
    const payload = jwt.verify(token, SECRET);
    req.user = payload;
    next();
  } catch (e) {
    return res.status(401).json({ message: 'Invalid token' });
  }
}

router.get('/jobs', async (req, res, next) => {
  try { await db.read(); res.json(db.data.jobs); } catch (err) { next(err); }
});

router.post('/jobs', auth, async (req, res, next) => {
  try {
    const { title, budget, image } = req.body;
    if (!title || !budget) return res.status(400).json({ message: 'Missing fields' });
    if (req.user.role !== 'client') return res.status(403).json({ message: 'Only clients can create jobs' });
    await db.read();
    const job = { id: Date.now().toString(), title, budget, image: image || null, status: 'open', applicants: [] };
    db.data.jobs.push(job);
    await db.write();
    res.json(job);
  } catch (err) { next(err); }
});

router.post('/jobs/:id/apply', auth, async (req, res, next) => {
  try {
    if (req.user.role !== 'freelancer') return res.status(403).json({ message: 'Only freelancers can apply' });
    await db.read();
    const job = db.data.jobs.find(j => j.id === req.params.id);
    if (!job) return res.status(404).json({ message: 'Job not found' });
    if (job.applicants.includes(req.user.id)) return res.status(400).json({ message: 'Already applied' });
    job.applicants.push(req.user.id);
    await db.write();
    res.json({ message: 'Applied' });
  } catch (err) { next(err); }
});

module.exports = router;
