const asyncHandler = require('express-async-handler');
const Person = require('../models/personNameModel');

/**
 * @desc    Get all person names
 * @route   GET /api/getAllPersonNames
 * @access  Private
 */
const getAllPersonNames = asyncHandler(async (req, res) => {
  // Fetch all person names from the database
  const allPersonNames = await Person.find({});

  res.status(200).json({ message: 'All Person Names retrieved!', personNames: allPersonNames });
});

/**
 * @desc    add new person name
 * @route   POST /api/addPersonName
 * @access  Private
 */
const addPersonName = asyncHandler(async (req, res) => {
  const { name } = req.body;

  // Create a new Person instance with the provided name
  const newPerson = new Person({ personName: name });

  // Save the new person to the database
  await newPerson.save();

  res.status(201).json({ message: 'Person Name added successfully!', data: newPerson });
});

/**
 * @desc    Delete person name by ID
 * @route   DELETE /api/deletePersonName/:id
 * @access  Private
 */
const deletePersonName = asyncHandler(async (req, res) => {
  const { id } = req.params;

  // Find the person by ID in the database
  const person = await Person.findById(id);

  // If the person is not found, return an error message
  if (!person) {
    return res.status(404).json({ message: 'Person not found' });
  }

  // Delete the person from the database
  await person.remove();

  res.status(200).json({ message: 'Person Name deleted successfully!', data: person });
});

module.exports = {
  getAllPersonNames,
  addPersonName,
  deletePersonName,
};
