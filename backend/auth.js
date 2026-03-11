const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { db, init } = require('../db');

const router = express.Router();
const SECRET = process.env.JWT_SECRET || 'dev-secret';

(async () => { await init(); })();

router.post('/register', async (req, res, next) => {
  try {
    const { name, email, password, role } = req.body;
    if (!name || !email || !password || !role) return res.status(400).json({ message: 'Missing fields' });
    await db.read();
    const exists = db.data.users.find(u => u.email === email);
    if (exists) return res.status(400).json({ message: 'User exists' });
    const hash = await bcrypt.hash(password, 10);
    const id = Date.now().toString();
    const user = { id, name, email, password: hash, role };
    db.data.users.push(user);
    await db.write();
    const token = jwt.sign({ id: user.id, email: user.email, role: user.role, name: user.name }, SECRET, { expiresIn: '7d' });
    console.log(`Registered user: ${email} as ${role}`);
    res.json({ token, role: user.role });
  } catch (err) { next(err); }
});

router.post('/login', async (req, res, next) => {
  try {
    const { email, password } = req.body;
    if (!email || !password) return res.status(400).json({ message: 'Missing fields' });
    await db.read();
    const user = db.data.users.find(u => u.email === email);
    if (!user) return res.status(400).json({ message: 'Invalid credentials' });
    const ok = await bcrypt.compare(password, user.password);
    if (!ok) return res.status(400).json({ message: 'Invalid credentials' });
    const token = jwt.sign({ id: user.id, email: user.email, role: user.role, name: user.name }, SECRET, { expiresIn: '7d' });
    console.log(`Login success: ${email}`);
    res.json({ token, role: user.role });
  } catch (err) { next(err); }
});

module.exports = router;
