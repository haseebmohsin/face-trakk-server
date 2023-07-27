const express = require('express');
const { liveVideo, getThumbnails } = require('../controllers/liveVideoController');
const router = express.Router();

// POST /api/videoUpload
router.post('/liveVideo', liveVideo);

// GET /api/videoUpload
router.get('/getThumbnails', getThumbnails);

module.exports = router;
