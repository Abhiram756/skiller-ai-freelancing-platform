const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    name: { type: String, required: true },
    role: { type: String, enum: ['student', 'client'], default: 'student' },
    company: String,
    skills: [String],
    wallet: { type: Number, default: 0 },
    isVerified: { type: Boolean, default: false }, // Track if initial face scan is done
    joinedAt: { type: Date, default: Date.now },

    // Skill Score System
    skillScore: { type: Number, default: 0 },
    skillBreakdown: {
        profile: { type: Number, default: 0 },
        skills: { type: Number, default: 0 },
        activity: { type: Number, default: 0 },
        projects: { type: Number, default: 0 },
        performance: { type: Number, default: 0 },
        market: { type: Number, default: 0 }
    },
    lastSkillScoreUpdate: { type: Date, default: Date.now },
    hiringFitScore: { type: Number, default: 0 } // Transient score for applications
});

module.exports = mongoose.model('User', UserSchema);
