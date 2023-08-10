const multer = require('multer');
const path = require('path');

// Create a storage engine for multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/videos/'); // Specify the destination folder
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const extname = path.extname(file.originalname);
    cb(null, 'video-' + uniqueSuffix + extname);
  },
});

// File upload middleware
const upload = multer({ storage });

module.exports = {
  upload,
};
