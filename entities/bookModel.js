// entities/bookModel.js
const mongoose = require('mongoose');

const bookSchema = new mongoose.Schema({
  title: String,
  description: String,
  discountRate: Number,
  coverImage: String,
  price: Number,
});

const Book = mongoose.model('Book', bookSchema);

module.exports = Book;
