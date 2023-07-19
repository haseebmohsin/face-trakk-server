const express = require('express');
const { videoUpload, getClustersData, getClusterDataById, xmlUpdate } = require('../controllers/videoUploadController');

const router = express.Router();

// POST /api/videoUpload
router.post('/upload', videoUpload);

// POST /api/xmlUpdate
router.post('/xmlUpdate', xmlUpdate);

// GET /api/getFaces
router.get('/getClustersData', getClustersData);

// GET /api/getClusterData/:id
router.get('/getClusterData/:id', getClusterDataById);

module.exports = router;
