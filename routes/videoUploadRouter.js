const express = require('express');
const { videoUpload, getFaces } = require('../controllers/videoUploadController');

const router = express.Router();

// POST /api/videoUpload
router.post('/upload', videoUpload);

// GET /api/getFaces
router.post('/upload', getFaces);

module.exports = router;
