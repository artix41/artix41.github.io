const mediumToMarkdown = require('medium-to-markdown');
 
mediumToMarkdown.convertFromUrl('<medium post url>')
.then(function (markdown) {
  console.log(markdown); //=> Markdown content of medium post
});
