const mongoose = require('mongoose');

const JobSchema = new mongoose.Schema({
    id: { type: String, unique: true },
    title: { type: String, required: true },
    budget: String,
    tags: [String],
    applicants: { type: Number, default: 0 },
    description: String,
    duration: String,
    university: String,
    postedAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('Job', JobSchema);
