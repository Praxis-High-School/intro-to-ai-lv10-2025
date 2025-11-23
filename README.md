# GitHub Collaboration Guide

## Overview

This guide explains how to collaborate on your capstone project using GitHub. **Each group will create ONE fork** of the shared repository, and all group members will work together on that fork. This teaches you real-world collaboration workflows.

**Main Repository**: https://github.com/Praxis-High-School/intro-to-ai-lv10-2025

## Repository Structure

```
Main Repository (Praxis-High-School/intro-to-ai-lv10-2025)
├── README.md (this file)
├── Group1/     (Group 1's workspace)
├── Group2/     (Group 2's workspace)
├── Group3/     (Group 3's workspace)
└── Group4/     (Group 4's workspace)
```

**Important**: Each group works in their own folder. Don't modify other groups' folders!

---

## Step 1: Fork the Repository (One Person Per Group)

**One person in your group** should fork the repository:

1. Go to: https://github.com/Praxis-High-School/intro-to-ai-lv10-2025
2. Click the **"Fork"** button (top right corner)
3. Choose where to fork (usually your personal GitHub account)
4. Click "Create fork"

**Result**: You now have your own copy of the repository (e.g., `your-username/intro-to-ai-lv10-2025`)

**Note**: Only ONE person per group needs to fork. Everyone else will work on that fork.

---

## Step 2: Add Group Members as Collaborators

**The person who forked** should add all group members:

1. Go to **your fork** (not the original repository)
2. Click **"Settings"** → **"Collaborators"**
3. Click **"Add people"**
4. Add all group members by their GitHub usernames
5. Everyone accepts the invitation (check your email or GitHub notifications)

**Result**: All group members can now push to the fork and collaborate together.

---

## Step 3: Clone Your Group's Fork

**Everyone in the group** should:

1. Go to **your group's fork** (the one created in Step 1)
2. Click the green **"Code"** button
3. Copy the repository URL (HTTPS)
4. Open terminal/command prompt
5. Run:
   ```bash
   git clone https://github.com/your-username/intro-to-ai-lv10-2025.git
   cd intro-to-ai-lv10-2025
   ```

**Replace `your-username`** with the GitHub username of the person who forked.

**Result**: You now have a local copy of your group's fork.

---

## Step 4: Navigate to Your Group Folder

After cloning, navigate to your group's folder:

```bash
# For Group 1
cd Group1

# For Group 2
cd Group2

# For Group 3
cd Group3

# For Group 4
cd Group4
```

**Remember**: Work only in your group's folder. Don't modify other groups' work!

---

## Step 5: Set Up Remote Tracking (Optional but Recommended)

To keep your fork updated with the main repository:

```bash
# Go back to repository root
cd ..

# Add the main repository as "upstream"
git remote add upstream https://github.com/Praxis-High-School/intro-to-ai-lv10-2025.git

# Verify remotes
git remote -v
```

You should see:
- `origin`: Your group's fork
- `upstream`: The main repository

---

## Step 6: Create a Branch for Your Work

**Before making changes**, create a branch:

```bash
# Make sure you're in the repository root
cd intro-to-ai-lv10-2025

# Navigate to your group folder
cd GroupX  # Replace X with your group number

# Create and switch to a new branch
git checkout -b your-name-work
# Example: git checkout -b john-notebook-section
```

**Why branches?**
- Keeps your work separate from others
- Allows multiple people to work simultaneously
- Makes it easier to review changes

---

## Step 7: Make Your Changes

1. Work on your notebook or files in your group folder
2. Save your changes
3. Test that everything works

**Remember**: 
- Work in your group folder only (Group1, Group2, Group3, or Group4)
- Don't modify files outside your group folder

---

## Step 8: Commit Your Changes

When you're ready to save your work:

```bash
# Make sure you're in the repository root (not just group folder)
cd /path/to/intro-to-ai-lv10-2025

# Add your changes
git add GroupX/  # Replace X with your group number
# Or add specific files
git add GroupX/capstone_notebook.ipynb
git add GroupX/README.md

# Commit with a message
git commit -m "Added data collection section to GroupX notebook"
```

**Good commit messages**:
- "Added model training section to Group1"
- "Updated Group2 README with setup instructions"
- "Fixed preprocessing code in Group3"
- "Added design thinking summary to Group4"

---

## Step 9: Push Your Changes

Push your branch to your group's fork:

```bash
git push origin your-name-work
```

**Result**: Your changes are now on GitHub in your group's fork.

---

## Step 10: Create a Pull Request to Main Repository

When you're ready to merge your work to the main repository:

1. Go to **your group's fork** on GitHub
2. You should see a banner: "your-name-work had recent pushes"
3. Click **"Compare & pull request"**
4. **Important**: 
   - Base repository: `Praxis-High-School/intro-to-ai-lv10-2025` (main branch)
   - Head repository: `your-username/intro-to-ai-lv10-2025` (your branch)
5. Write a description:
   - What did you add/change?
   - Which group folder?
   - Any important notes?
6. Click **"Create pull request"**
7. Wait for instructor review and merge

**Result**: Your work will be merged into the main repository after review.

---

## Step 11: Update Your Local Copy

**After someone merges changes** (or to get updates from main repository):

```bash
# Switch to main branch
git checkout main

# Pull latest changes from your fork
git pull origin main

# (Optional) Get updates from main repository
git fetch upstream
git merge upstream/main
```

**Result**: Your local copy is up to date.

---

## Collaboration Best Practices

### ✅ Do's

- **Communicate**: Tell your group what you're working on
- **Pull before push**: Always pull latest changes before pushing
- **Use branches**: Don't work directly on main branch
- **Commit often**: Save your work regularly
- **Write clear messages**: Explain what you changed
- **Test before commit**: Make sure your code works
- **Work in your group folder**: Don't touch other groups' folders

### ❌ Don'ts

- **Don't commit API keys**: Use environment variables
- **Don't delete others' work**: Be careful with git commands
- **Don't work on same file simultaneously**: Coordinate with your group
- **Don't force push**: Can overwrite others' work
- **Don't commit large files**: Use Git LFS if needed
- **Don't modify other groups' folders**: Stay in your own folder

---

## Resolving Conflicts

If you get a **merge conflict**:

1. Don't panic! This is normal when multiple people edit the same file
2. Git will mark the conflicts in the file
3. Open the file and look for conflict markers:
   ```
   <<<<<<< HEAD
   Your changes
   =======
   Their changes
   >>>>>>> branch-name
   ```
4. Decide which version to keep (or combine both)
5. Remove the conflict markers
6. Save the file
7. Commit the resolved file:
   ```bash
   git add filename
   git commit -m "Resolved merge conflict"
   ```

---

## Common Git Commands

```bash
# Check status
git status

# See what changed
git diff

# See commit history
git log

# Switch branches
git checkout branch-name

# Create new branch
git checkout -b new-branch-name

# See all branches
git branch

# Delete a branch (after merging)
git branch -d branch-name

# See remote repositories
git remote -v

# Update from upstream (main repository)
git fetch upstream
git merge upstream/main
```

---

## Workflow Summary

**Daily workflow for group members**:

```bash
# 1. Get latest changes
cd intro-to-ai-lv10-2025
git checkout main
git pull origin main

# 2. Create branch for your work
git checkout -b my-feature

# 3. Navigate to your group folder
cd GroupX  # Your group number

# 4. Make changes
# ... edit files ...

# 5. Commit changes
cd ..  # Back to repo root
git add GroupX/
git commit -m "Description of changes"

# 6. Push to your group's fork
git push origin my-feature

# 7. Create pull request on GitHub (from your fork to main repo)
```

---

## Getting Help

If you're stuck:

1. **GitHub Help**: https://docs.github.com
2. **Git Tutorial**: https://git-scm.com/docs
3. **Ask your group**: Someone might know the answer
4. **Ask instructor**: During mentoring session or class

---

## Final Submission Checklist

Before Session 10, make sure:

- [ ] Your group has forked the repository
- [ ] All group members are added as collaborators to the fork
- [ ] All notebook sections are complete in your group folder
- [ ] README.md is in your group folder with setup instructions
- [ ] All code is committed and pushed to your group's fork
- [ ] Pull request created from your fork to main repository
- [ ] Pull request is merged (or ready for review)
- [ ] No API keys are in the code
- [ ] All group members have contributed
- [ ] Your group folder contains: `capstone_notebook.ipynb` and `README.md`
- [ ] Repository link is ready for presentation: https://github.com/Praxis-High-School/intro-to-ai-lv10-2025

---

## Quick Reference

**Repository Links**:
- Main Repository: https://github.com/Praxis-High-School/intro-to-ai-lv10-2025
- Your Group's Fork: `https://github.com/your-username/intro-to-ai-lv10-2025`

**Key Points**:
- One fork per group (not per person)
- All group members work on the same fork
- Create branches for individual work
- Use pull requests to merge to main repository
- Work only in your group folder

**Good luck with your collaboration! 🚀**
