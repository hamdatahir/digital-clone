<!-- Long-form document (500-1500 words): a write-up of ONE real project, e.g. your FYP or a course project. Use real names of tools, models, datasets. -->

# Project Writeup: CampusConnect

## The Problem
At UMT, students often struggle to find study materials like past papers, helpful notes, and reference books. Currently, these resources are scattered across different WhatsApp groups, Google Drives, and word of mouth. Students waste a lot of time searching for materials that they know exist but can't locate. I wanted to build a centralized platform where students could easily upload and download study resources.

## How It Works
CampusConnect is a web application built using Python and Flask. When a student creates an account, they can browse resources by course code or semester. If they want to upload a file, they select the course, upload the PDF or image, and add a short description. The file is then stored on the server, and the metadata is saved in a MySQL database. Other students can search for the course code and download the file.

## My Role
This was a solo project, so I handled everything from start to finish. I designed the database schema, wrote the backend logic in Flask, created the HTML/CSS templates, and handled the file upload system. 

## Challenges
The biggest challenge I faced was handling file uploads securely. Initially, my code allowed anyone to upload any file type, which was a security risk. I had to implement a validation system that checks the file extension and limits the maximum file size. I also had to learn how to handle the database connections properly. At first, the app would crash if multiple people tried to upload files at the same time because I wasn't closing the database sessions. I fixed this by learning how to use context managers and connection pools in Flask.

## Results
The platform was tested with a group of about 15 classmates. They found it very useful, and it saved them a lot of time. I learned a tremendous amount about backend development and how to handle real-world deployment issues.
