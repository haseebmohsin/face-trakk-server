const express = require('express');
const { videoUpload, getData } = require('../controllers/videoUploadController');

const router = express.Router();

// POST /api/videoUpload
router.post('/upload', videoUpload);

// GET /api/getFaces
router.get('/getData', getData);

module.exports = router;
