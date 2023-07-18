const express = require('express');
const { videoUpload, getClustersData, getClusterDataById } = require('../controllers/videoUploadController');

const router = express.Router();

// POST /api/videoUpload
router.post('/upload', videoUpload);

// GET /api/getFaces
router.get('/getClustersData', getClustersData);

// GET /api/getClusterData/:id
router.get('/getClusterData/:id', getClusterDataById);

module.exports = router;
