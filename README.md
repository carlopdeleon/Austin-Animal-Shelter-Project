# Austin-Animal-Shelter-Project

# Project Description
This project is to delve into Austin's Animal Shelter statistics to see how an animal was onboarded and what that particular animal's outcome was. With the data acquired, it can be used to discover trends in what influences an animal's outcome. With the trends discovered, it could be possibly used to strategically produce a program that could increase the chances of an animal having a positive outcome.

# Initial Thoughts
My intitial thoughts on this project is that an animal's outcome will be heavily influenced by their age and breed. Their color will have a small influence on their chances at being adopted. I also wanted to explore if their time in shelter somehow made them less desireable to potential adoptive parents. 

# Acquire




# Prepare
- Normalized columns by making them pythonic
- Merged intake and outcome dataframes
    - 190,609 rows after merge
- Dropped nulls
    - 190,511 rows after drop
- Subset dataframe for only dogs
    - 119,868 rows for dogs
- Dropped columns:
    - found location
    - outcome_subtype
    - breed_x
    - animal_type_y
    - color_x
    - monthyear_x
    - monthyear_y
    
- Drop values that are Rto-Adopt, Disposal, Missing, Stolen, Transfer
- Created features using existing data
    - time in shelter
    - dropped any rows that had negative days. 48873 rows dropped.
    - age_outcome: age they were upon their outcome
- Simplified Top 10 breeds to represent majority breed
    - over 2000 breeds
 


# Explore
**Questions**
1. Does the animal's breed influence their chances of being placed in a home?
2. Does their color influence their chances of being placed in a home?
3. Does their time in a shelter decrease their desirability to potential adoptive parents?
4. Does their age influence their chances of being adopted?