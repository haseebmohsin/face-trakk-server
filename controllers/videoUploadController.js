const multer = require('multer');
const path = require('path');

// Create a storage engine for multer
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const extname = path.extname(file.originalname);
    cb(null, 'video-' + uniqueSuffix + extname);
  },
});

// Create multer instance with the storage engine
const upload = multer({ storage });

/**
 * @desc    Upload Video for face recognition
 * @route   POST /api/videoUpload
 * @access  Private
 */
const videoUpload = (req, res) => {
  upload.single('video')(req, res, (err) => {
    if (err) {
      // Handle any error that occurs during the upload
      console.error('Error uploading file:', err);
      return res.status(500).json({ error: 'Error uploading file' });
    }

    // File upload is successful
    res.status(200).json({ message: 'Video uploaded successfully' });
  });
};

module.exports = {
  videoUpload,
};
