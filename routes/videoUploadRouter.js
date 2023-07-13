const express = require('express');
const { videoUpload } = require('../controllers/videoUploadController');

const router = express.Router();

// POST /api/videoUpload
router.post('/upload', videoUpload);

module.exports = router;
