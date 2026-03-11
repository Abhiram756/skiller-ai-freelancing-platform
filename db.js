const mongoose = require('mongoose');

// Disable buffering so it fails fast when offline
mongoose.set('bufferCommands', false);

const connectDB = async () => {
  try {
    // REVERTING TO REMOTE DB AS REQUESTED
    console.log("Attempting to connect to REMOTE MongoDB Atlas...");

    const conn = await mongoose.connect(process.env.MONGO_URI, {
      family: 4, // Force IPv4
      serverSelectionTimeoutMS: 10000 // Give Atlas 10 seconds to respond
    });
    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (err) {
    console.error(`MongoDB Connection Error: ${err.message}`);
    console.log("⚠️  Warning: Running in Offline/Limited Mode (Database not connected)");
    // process.exit(1); // Don't crash the server, let it serve static files
  }
};

module.exports = connectDB;