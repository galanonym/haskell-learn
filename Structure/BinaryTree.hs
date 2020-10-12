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
