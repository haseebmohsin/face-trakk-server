const express = require('express');
const router = express.Router();
const { getAllPersonNames, addPersonName, deletePersonName } = require('../controllers/personController');

// GET /api/getAllPersonNames
router.get('/getAllPersonNames', getAllPersonNames);

// POST /api/addPersonName
router.post('/addPersonName', addPersonName);

// DELETE /api/deletePersonName/:id
router.delete('/deletePersonName/:id', deletePersonName);

module.exports = router;
