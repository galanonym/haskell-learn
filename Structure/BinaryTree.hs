module Structure.BinaryTree (treeInsert, treeElem, exampleTree, Tree(..)) where -- exports everything
-- module Structure.BinaryTree (treeInsert, treeElem) where -- exports only functions not constructors

-- a tree is either an empty tree or an alament that contains a value and two trees
-- EmptyTree and Node are value constructors
data Tree a = EmptyTree | Node a (Tree a) (Tree a) deriving (Show) 

-- utitly to make a tree with just one node (tree branch endings)
singleton :: a -> Tree a
singleton x = Node x EmptyTree EmptyTree

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert x EmptyTree = singleton x -- if we reach EmptyTree that means value x has come to tree branch ending
treeInsert x (Node value leftTree rightTree)
  | x == value = Node x leftTree rightTree -- if element x we want to insert is allready in tree just return the tree
  | x < value = Node value (treeInsert x leftTree) rightTree -- same root value, same rightTree but insert x element to leftTree
  | x > value = Node value leftTree (treeInsert x rightTree)

treeElem :: (Ord a) => a -> Tree a -> Bool
treeElem x EmptyTree = False
treeElem x (Node value leftTree rightTree)
  | x == value = True
  | x < value = treeElem x leftTree
  | x > value = treeElem x rightTree

-- testing
exampleTree = foldl (flip treeInsert) EmptyTree [4,6,8,1,7,3,5]
{-

Node 4 
  (Node 1 
    EmptyTree 
    (Node 3 EmptyTree EmptyTree)
  ) 
  (Node 6 
    (Node 5 EmptyTree  EmptyTree) 
    (Node 8 
      (Node 7 EmptyTree EmptyTree) 
      EmptyTree
    )
  )

-}

is7 = treeElem 7 exampleTree
is9 = treeElem 9 exampleTree

instance Functor Tree where
  fmap f EmptyTree = EmptyTree
  fmap f (Node value leftTree rightTree) = Node (f value) (fmap f leftTree) (fmap f rightTree)
  -- our function applied to first node value, and then applied recursivly to leftTree and rightTree

exampleTreeMulti = fmap (*2) exampleTree
-- Node 8 (Node 2 EmptyTree (Node 6 EmptyTree EmptyTree)) (Node 12 (Node 10 EmptyTree EmptyTree) (Node 16 (Node 14 EmptyTree EmptyTree) EmptyTree))

-- !Warning - after fmapping the tree may not be longer a binary search tree, just normal binary tree, because not following rule where rightTree has values smaller then value and leftTree bigger that value
