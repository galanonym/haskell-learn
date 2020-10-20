-- import must be on top

---- Build in modules

import Debug.Trace (trace) -- only import trace function
import Data.List
import Data.Char
import qualified Data.Map as Map -- Map functions clash with Prelude

---- User defined modules

import qualified Geometry.Sphere as Sphere -- both have function "area" defined
import qualified Geometry.Cuboid as Cuboid
import qualified Geometry.Cuboid as Cuboid
-- > Cuboid.area 1 2 3 -- 22.0 

import qualified Structure.BinaryTree as BinaryTree


---- Built in types

-- Int -- bounded integer 2^63
-- Integer -- unbounded integer
-- Float -- floating point
-- Double -- float with more precision
-- Bool -- boolean
-- Char -- single character
-- String -- list of single characters
-- [Char] -- list of single characters

-- type classes gives ability

-- Eq -- testing for equality
-- Ord -- testing for ordering < >
-- Show -- printable as strings
-- Read -- convert from sting to type
-- Enum -- sequentially order
-- Bounded -- has a min / max value
-- Num -- can act as numbers (Int, Float etc.)
-- Floating -- can act as floats
-- Fractional -- can act as floats and fractions
-- Integral -- can act as ints (Int, Integer)

-- :: means "Has a type of"
-- -> means "Returns"
-- (Eq a) class constraint
-- Eq a => apply type class Eq to type variable a

---- Built in functions

-- > succ 5 -- 6 -- next successor
-- > min 5 4 -- 4 
-- > max 5 4 -- 5
-- > div 4 2 -- 2 -- integral division
-- > subtract 5 4 -- -1 -- subtracts 5 from 4
-- > odd 3 -- True
-- > even 4 -- True
-- > negate -5 -- 5 -- change sign of number
-- > abs -4 -- 4 -- absolute value
-- > ceiling 1.09 -- 2
-- > floor 1.89 -- 1
-- > round 1.89 -- 2
-- > id True -- True -- just return argument

-- > head [5,4,3] -- 5
-- > tail [5,4,3] -- [4,3] -- chops off head
-- > last [5,4,3] -- 3
-- > init [5,4,3] -- [5,4] -- everything but last element
-- > length [5,4,3] -- 3
-- > null [5,4,3] -- False -- is empty
-- > null [] -- True
-- > reverse [5,4,3] -- [3,4,5]
-- > maximum [5,4,3] -- 5
-- > minimum [5,4,3] -- 3
-- > take 2 [5,4,3] -- [5,4] -- returns first 2 elements
-- > drop 2 [5,4,3] -- [3] -- drops first 2 elements
-- > sum [5,4,3] -- 12 
-- > product [5,4,3] -- 60
-- > elem 5 [5,4,3] -- True -- is element present in list
-- > (!!) [5,4,3] 2 -- 3 -- find 2th element (from zero)
-- > cycle [5,4,3] -- [5,4,3,5,4,3,5,4,3 ... (to infinity)]
-- > repeat 5 -- [5,5,5,5 ... (to infinity)]
-- > replicate 3 5 -- [5, 5, 5]
-- > concat [[2,3], [1,2]] -- [2,3,1,2] -- concatenates lists of lists
-- > and [True, True, True] -- True
-- > and [True, False, True] -- False
-- > or [True, False, True] -- True
-- > or [False, False, False] -- False
-- > (&&) True False -- False
-- > (||) True False -- True

-- > fst (5,4) -- 5 -- only pairs
-- > snd (5,4) -- 4 -- only pairs
-- > zip [5,4,3] ['a', 'b', 'c'] -- [(5, 'a'), (4, 'b'), (3, 'c')] -- zips elements of two lists into pairs
-- > zipWith (+) [5,4,3] [1,2,3] -- [6,6,6]

-- > compare 2 4 -- LT -- checks for equality, returns GT LT EQ of Ordering type
-- > show 5 -- "5" -- convert value as string
-- > read "True" :: Bool -- True -- convert a string, type must be supplied

-- > flip replicate 3 5 -- [3,3,3,3,3] -- flips next function arguments
-- > flip (:) [2] 1 -- [1,2] -- flips : so last argument is added to beginning of list
-- > takeWhile (/=' ') "im a boss"  -- "im" -- returns list of elements as long as predicate holds true

-- > map (+3) [5,4,3] -- [8, 7, 6] -- applies function to each element of list, returns new list
-- > filter (>3) [1,4,5,6,3,2] -- [4,5,6] -- returns the list of those elements that satisfy the predicate
-- > all (>3) [4,5,6] -- True -- determines if all elements in structutre (list) satisfy the predicate

-- > foldl (\acc x -> acc + x) 0 [1,2,3] -- 6 -- takes binary function, starting accumulator and list, applies function from left to each element accumulationg
-- > foldr (\x acc -> x : acc) [] [1,2,3] -- [3,2,1] -- takes binary function, starting accumulator and list, applies function from right to each element accumulationg
-- > foldl1 -- assumes first element from left as accumulator and moves to next
-- > foldr1 -- assumes first element from right as accumulator and moves to next
-- > scanl, scanr, scanl1, scanr1 -- same as fold's, but report all intermediate accumulator states in a list, used to debug scans

-- > const 5 8 -- 5 -- always returns first argument, used when we want to replace arguments in function composition with a constant value so -- > map (const 42) [1..5] -- [42,42,42,42,42]

---- Data.List functions

-- > nub [1,1,2,3,3] -- [1,2,3] -- only unique list elements
-- > words "hey you boss" -- ["hey", "you", "boss"] -- splits string into list of words by whitespace
-- > unwords ["hey", "you", "boss"] -- "hey you boss" -- joins list of words into string with space
-- > group [1,1,2,3,3,1] -- [[1,1], [2], [3,3], [1]] -- groups adjacent elements
-- > sort [1,3,2] -- [1,2,3]
-- > tails [1,2,3] -- [[1,2,3], [2,3], [3], []] -- takes list and sucessfully applies tails to it
-- > isPrefixOf [1,2] [1,2,3] -- True -- if second list starts with the first one
-- > any (>4) [1,2,3] -- False -- takes predicate function and tells if any element in list satisfies it
-- > isInfixOf [2,3] [1,2,3,4] -- True -- if sectond list includes first one
-- > foldl' - stricter version of foldl that does not stack overflow
-- > find (>4) [3,4,5,6] -- Just 5 -- stops after finding element matching predicate, returns Maybe type

---- Maybe

-- > maybe "Default value" show $ Just 5 -- unwraps Just 5 through show function, displays "Default value" if Nothing 

---- Data.Char functions

-- > ord 'a' -- 97 -- numeric representation for char from unicode table
-- > chr 97 -- 'a' -- convert numeric representation to char
-- > digitToInt '3' -- 3 -- convert numeric character to corresponding integer 
-- > isDigit '3' -- True -- returns true if character is a digit

---- Data.Map functions

-- > Map.fromList [("AB", 1), ("CD", 1)] -- convert list of pairs to map type -- when printed it prints in fromList form
-- > Map.lookup "betty" phoneBook -- Just "555-340-123" -- finds value by key in map, returns Maybe
-- > Map.insert "grace" "341-5400" phoneBook -- returns new Map with new entry
-- > Map.size phoneBook -- 3 -- returns size of the Map
-- > Map.map (0 ++)

---- Functions

doubleMe x = x + x -- function name, parameters, code that makes body of function

doubleUs x y = doubleMe x + doubleMe y
x `doubleUsInfix` y = doubleMe x + doubleMe y -- infix functions

-- assigning special character function
(&&&) :: Bool -> Bool -> Bool
True &&& x = x
False &&& _ = False

---- if..then..else
doubleSmallNumber x = if x > 100
  then x
  else x * 2 -- else is mandatory

doubleSmallNumber' x = (if x >100 then x else x * 2) + 1

---- Lists

lostNumbers = [4,8,15,16,23,42]

lostNumbers' = lostNumbers ++ [99] -- append

smallCat = 'A' : " SMALL CAT" -- prepend

letterB = "Steve Buscemi" !! 6 -- find 6th element
letterB' = (!!) "Steve Buscemi" 6 -- as prefix not infix

big13number = [13,26..] !! 124 -- infinite lists

---- List comprehensions

filterSpecial :: [Char] -> [Char]
filterSpecial string = [ character | character  <- string, character `elem` '_':['0'..'9'] ++ ['a'..'z'] ++ ['A'..'Z'] ]
-- before | -- how we want elements to be reflected in resulting list
-- number <- numbers -- we draw elements from this list
-- , -- predicate/condition, one or more

boomBangs numbers = [ if number < 10 then "Boom!" else "Bang!" | number <- numbers, odd number ]

-- multiple lists
multipleTuples :: [(Int, Char)]
multipleTuples = [(n, c) | n <- [1,2,3], c <- ['a','b']]
-- [(1,'a'),(1,'b'),(2,'a'),(2,'b'),(3,'a'),(3,'b')]

-- Triangles example:
-- Find right triangle that has sides are integers, length of each side is 0<x<10, sum of all sides is 24
-- generate all possible triples with elements less or equal 10
triples = [ (a,b,c) | c <- [1..10], a <- [1..10], b <- [1..10] ] 
triples' = [ (a,b,c) | c <- [1..10], a <- [1..c], b <- [1..a] ] 

-- generate all possible triples with elements less or equal 10
rightAngles = [ (a,b,c) | c <- [1..10], a <- [1..10], b <- [1..10], a^2 + b^2 == c^2 ] 
rightAngles' = [(a,b,c) | (a,b,c) <- triples', a^2 + b^2 == c^2]

-- apply last condition
rightTriangles = [(a,b,c) | (a,b,c) <- rightAngles', a + b + c == 24]

---- Tuples

exampleTuple = (3, 1.1, 'a', "hello") -- hold different element types

zigzagTuples = zip [1,2,3] ['a', 'b', 'c'] -- create tuple from lists
zigzagTuples' = zip [1..] ['a', 'b', 'c']

---- Types
circumference :: Float -> Float
circumference r = 2 * pi * r

circumference' :: Double -> Double -- More precision
circumference' r = 2 * pi * r

whatIsGreater = "abracadabra" `compare` "zebra" -- LT, GT or EQ? Will return ordering type variable
whatIsGreater' = show whatIsGreater -- Convert to string

---- Pattern matching

-- for different function bodies
errorMessage :: Int -> String
errorMessage 404 = "Not Found"
errorMessage 403 = "Forbidden"
errorMessage 501 = "Internal Server Error"
errorMessage i = "Other Error"

-- in tuples
third :: (a, b, c) -> c
third (_, _, z) = z

-- pattern inside list comprehension
listOfSumsOfPairs = [a + b | (a, b) <- [(1, 3), (4, 3), (2, 4), (5, 3)]] -- (a, b) is a pattern matched when drawing elements from list

-- pattern for list, our own head implementation
head' :: [a] -> a
head' [] = error "Can't call head on empty list" -- error function dies execution
head' (x:_) = x -- x is first element, _ is rest of the list, paranteses used when binding to several variables (not a tuple)

---- As-pattern

firstLetter :: String -> String
firstLetter "" = "Empty string, woops!"
firstLetter all@(x:xs) = "First letter is of string " ++ all ++ " is " ++ [x]

---- Guards and where

-- can be used after pattern matching line
bmiTell :: Float -> Float -> String
bmiTell weight height -- no equal sign when using guards
  | bmi <= skinny = "Underweight! " ++ bmiText -- guard is an if statement that falls through on false
  | bmi <= normal = "Looking good! " ++ bmiText 
  | bmi <= big = "Overweight. " ++ bmiText
  | otherwise = "Fat. " ++ bmiText -- otherwise special keyword
  where bmi = weight / height ^ 2
        bmiText = "BMI: " ++ take 5 (show bmi)
        (skinny, normal, big) = (18.5, 25, 30) -- can use pattern matching, variable is drawn from tuple

-- take list of weight and heights and return list of bmi's
bmiBulk :: [(Float, Float)] -> [Float]
bmiBulk listOfPairs = [ calculate weight height | (weight, height) <- listOfPairs ]
  where calculate weight height = weight / height ^ 2 -- function inside where statement

---- let..in

-- can be used in other expressions
-- things after let are locally scoped to expression after in
squares = [let square x = x * x; twoMore = 2 in (square 5 + twoMore, square 4 + twoMore)] -- [(27, 18)]

-- let..in used in list comprehension
-- local variable avaiable in output and predicates not generator
bmiBulk' :: [(Float, Float)] -> [Float]
bmiBulk' listOfPairs = [bmi | (w, h) <- listOfPairs, let bmi = w / h ^ 2]

---- case..of

-- can be used as experssion
-- similar to switch statement
describeList :: [a] -> String
describeList list = "This list is " ++ case list of 
  [] -> "empty."
  [x] -> "singleton."
  other -> "longer."

---- Recursion

-- returns highest of a list
-- debug statement from https://stackoverflow.com/a/9849243
maximum' :: (Ord a, Show a) => [a] -> a
maximum' [] = error "Cannot get maximum of empty list"
maximum' [x] = trace ("maximum' [" ++ show x ++ "] == " ++ show x) $ x
maximum' (x:xs) = trace ("max " ++ show x ++ " (maximum' " ++ show xs ++ ")") $ max x (maximum' xs) 
-- split list into head x and tail (rest) xs
-- max -- returns the larger of its two arguments
-- debugging, add at top -- import Debug.Trace
-- trace <string> -- print to console
-- trace (show x) $ x -- equivalent to -- (trace (show x)) (x)

-- Example:
-- maximum' [2,5,1]
-- First two patterns don't match
-- Split list: max 2 (maximum' [5,1]) 
-- Split list: max 2 (max 5 (maximum' [1]))
-- (maximum' [1]) -- returns 1
-- Values are: max 2 (max 5 1)
-- (max 5 1) -- returns 5
-- Values are: max 2 5
-- (max 2 5) -- returns 5


-- returns highest of a list (without using max), my own implementation
maximum'' :: (Ord a, Show a) => [a] -> a
maximum'' [] = error "Cannot get maximum of empty list"
maximum'' [x] = x
maximum'' (x:xs)
  | x > maximum'' xs = x
  | otherwise = maximum'' xs 

-- example
-- maximum'' [2,5,1]
  -- first two patterns don't match
  -- first guard check: 2 > maximum'' [5, 1] 
    -- next recursion: maximum'' [5, 1] 
    -- first guard check: 5 > maximum [1]
    -- 5 > 1
    -- true, return 5
  -- back a level: 2 > 5
  -- first guard check: False
  -- second guard check:
  -- maximum'' [5] 
  -- which is 5

-- replicates value x times in a list
replicate' :: Int -> a -> [a]
replicate' times value 
  | times <= 0 = []
  | otherwise = value : replicate (times-1) value

-- returns first n elements from provided list
take' :: Int -> [a] -> [a]
take' n _
  | n <= 0 = []
take' _ [] = []
take' n (x:xs) = x : (take' (n-1) xs)

-- reverse a list
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = (reverse' xs) ++ [x]

-- zip a list into pairs. 
-- ex. zip [1,2,3] [7,8] -- [(1,7), (2,8)]
zip' :: [a] -> [b] -> [(a, b)] 
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x, y) : zip' xs ys

-- checks if element is in the list
elem' :: Eq a => a -> [a] -> Bool
elem' _ [] = False
elem' y (x:xs)
  | x == y = True
  | otherwise = elem' y xs

-- quicksort
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
  let smallerOrEqual = [n | n <- xs, n <= x ];
      bigger = [n | n <- xs, n > x]
  in quicksort smallerOrEqual ++ [x] ++ quicksort bigger

---- Partially applied functions

-- A curried function is a function that takes multiple arguments one at a time. 

-- every function takes one parameter
multipleThreeNumbers :: Int -> Int -> Int -> Int
multipleThreeNumbers x y z = x * y * z
multipleTwoNumbersWithNine = multipleThreeNumbers 9
-- it partially applies 9 as parameter, is equivalent to:
multipleTwoNumbersWithNine' y z = 9 * y * z

---- Sectioning

sumOfTwoAndThree = (+) 2 3 -- plus as prefix, is equivalent to:
sumOfTwoAndThree' = (+2) 3 -- sectioning of plus function

addTwo :: Num a => a -> a
addTwo = (+2) -- partially applied + functio

appendLol :: String -> String 
appendLol = (++ " LOL") -- partially applied ++ function

prependLol :: String -> String
prependLol = ("LOL " ++) -- infix functions can be sectioned from behind

addAtBeginning :: Num a => [a] -> [a]
addAtBeginning = (3:)

isUpperAlphanum :: Char -> Bool
isUpperAlphanum = (`elem` ['A'..'Z']) -- elem as infix function so second param is partially applied

-- (-3) means negative value, use (subtract 3) when partially applying minus
subtractTwoFromFive = subtract 2 5 -- subtract 2 from 5

---- Higher order functions

applyTwice :: (a -> a) -> a -> a -- first param is a function
applyTwice f x = f (f x)
-- > applyTwice (+3) 10 -- 16
-- > applyTwice (++ " HAHA") "HEY" -- "HEY HAHA HAHA"

-- similar to zip' function, but takes joining function
zipWith' :: (a -> b -> c) -> [a] -> [b] -> [c]
zipWith' _ [] _ = []
zipWith' _ _ [] = []
zipWith' f (x:xs) (y:ys) = f x y : zipWith' f xs ys

zipper :: Num a => a -> a -> a
zipper x y = x + y + 1
-- > zipWith' zipper [6,2,3] [1,1,2] -- [8,4,6]
-- > zipWith' (*) (replicate 5 2) [1..] -- [2,4,6,8,10]

flip' :: (a -> b -> c) -> (b -> a -> c) -- takes a function returns a function
flip' f = g
  where g x y = f y x
flip'' f x y = f y x -- we can decompose first function as pattern

map' :: (a -> b) -> [a] -> [b]
map' _ [] = []
map' f (x:xs) = f x : map' f xs
-- > map' (+3) [1,5,3] -- [4,8,6]
-- > map' fst [(1,2),(3,5),(6,1)] -- [1,3,6]

filter' :: (a -> Bool) -> [a] -> [a]
filter' _ [] = []
filter' predicate (x:xs) 
  | predicate x = x : filter' predicate xs
  | otherwise = filter' predicate xs
-- > filter' (>3) [1,4,5,6,3,2] -- [4,5,6]
-- > filter' even [1,4,5,6,3,2] -- [4,6,2]
-- > filter' (`elem` ['A'..'Z']) "I am The Boss" -- "ITB"

-- better quicksort using filter function
quicksort' :: Ord a => [a] -> [a]
quicksort' [] = []
quicksort' (x:xs) = quicksort' (filter' (<= x) xs) ++ [x] ++ quicksort' (filter' (> x) xs)

collaz :: Int -> [Int]
collaz 1 = [1]
collaz i
  | even i = i : collaz (i `div` 2)
  | odd i = i : collaz (i * 3 + 1)

-- for numbers 1 to 100 how many collaz chains have length grater then 15?
numLongChains :: Int
numLongChains = length (filter (>15) (map length (map collaz [1..100])))

-- lambda pattern matching
onlyHeads = map (\(x:xs) -> x) [[1,2,3], [2,3,4]]

-- foldl applies binary function to accumulator and left value of the list and so on returns accumulator
sum' :: Num a => [a] -> a
sum' xs = foldl (\acc x -> acc + x) 0 xs -- takes binary function, starting accumulator, list
sum'' :: Num a => [a] -> a -- needs type annotation or error
sum'' = foldl (+) 0 -- returns a function that takes list as first argument

-- foldr takes values from the list from right to left
-- used when building a new list from a list
map'' :: (a -> b) -> [a] -> [b]
map'' f xs = foldr (\x acc -> f x : acc ) [] xs -- reversed order of arguments in binary function
-- > map'' (+3) [1,2,3] -- [4,5,6]

elem'' :: Eq a => a -> [a] -> Bool
elem'' e xs = foldr (\x acc -> (if x == e then True else acc)) False xs

filter'' :: (a -> Bool) -> [a] -> [a]
filter'' p = foldr (\x acc -> (if p x then x : acc else acc)) [] -- returns a function that takes list as first argument, p = predicate

-- foldr f z [3,4,5,6] -- is equivalent to
-- f 3 (f 4 (f 5 (f 6 z)))

-- foldl f z [3,4,5,6] -- is equivalent to
-- f (f (f (f z 3) 4) 5) 6)

---- Functional application operator $

-- when $, the expression on right is becoming parameter the function on the left

bigResult :: Int
bigResult = sum (filter (>10) (map (*2) [2..10])) -- 80
bigResult' = sum $ filter (>10) $ map (*2) [2..10] -- 80
-- f $ g $ j <==> f $ (g $ j)

-- $ is a normal function that can be mapped
calculations :: [Float]
calculations = map ($ 3) [(4+), (10*), (^2), sqrt] -- [7.0, 30.0, 9.0, 1.7320508]

---- Functional composition operator .

makeAllNegative :: Num a => [a] -> [a]
makeAllNegative = map (\x -> negate $ abs x) -- returns function that needs list as first argument
makeAllNegative' = map (negate . abs) -- number is abs then it is negated
makeAllNegative'' = map $ negate . abs -- functional application can be used in this case because only one argument is needed

biggerExample :: (Num a, Ord a) => [a]
biggerExample = replicate 2 (product (map (*3) (zipWith max [1,5] [4,2])))
biggerExample' = replicate 2 . product . map (*3) $ zipWith max [1,5] [4,2] 

---- Better error messages

-- > let res = negate . abs 5 
-- returns a cryptic error merrage
-- try with type annotation, what do we expect res to be?
-- > let res :: Num a => a; res = negate . abs 5
-- much better, indicates problem with (.) too many arguments
-- :set -XTypeApplications (https://www.reddit.com/r/haskell/comments/iydvze/tips_on_how_to_make_sense_of_haskell_errors/g6c7al6/)

---- Solving problems with module functions

-- Make touples with word count in string
-- ex. countingWords ["wa wa wee wa"] -- [(wa, 3), (wee, 1)]
countingWords :: String -> [(String, Int)]
countingWords string = map (\words -> (head words, length words)) $ (group . sort . words) string
countingWords' = map (\words -> (head words, length words)) . group . sort . words -- make it point free style

-- Check if list [3,4] is contained in [1,2,3,4,5]

isInList :: Eq a => [a] -> [a] -> Bool 
isInList needle haystack = any (==True) $ map (\part -> isPrefixOf needle part) $ tails haystack
isInList' needle = any (==True) . map (\part -> isPrefixOf needle part) . tails -- make it point free style
isInList'' needle = any (==True) . map (isPrefixOf needle) . tails -- lambda can be omittend 
isInList''' needle = any (isPrefixOf needle) . tails -- any works like map
isInList'''' needle = any (needle `isPrefixOf`) . tails -- more readable as infix function

-- Encode message in ceasar cipther

encodeCeasar :: Int -> [Char] -> [Char]
encodeCeasar offset message = map chr $ map (+ offset) $ map ord message -- ord returns unicode number for a character
encodeCeasar' offset = map chr . map (+ offset) . map ord -- point free style
encodeCeasar'' offset = map (chr . (+ offset) . ord) -- one map only
encodeCeasar''' offset = map $ chr . (+ offset) . ord -- functional application

decodeCeasar :: Int -> [Char] -> [Char]
decodeCeasar offset message = encodeCeasar (negate offset) message
decodeCeasar' offset = encodeCeasar $ negate offset -- without message
decodeCeasar'' = encodeCeasar . negate -- point free style

-- First natural numbert with sum of its digits equials 40

-- Subfunction
digitSum :: Int -> Int 
digitSum = sum . map (\c -> read [c] :: Int) . show
digitSum' :: Int -> Int
digitSum' = sum . map digitToInt . show

firstTo40 :: Maybe Int 
firstTo40 = find (\n -> digitSum n == 40) [1..]
firstTo40' = find ((==40) . digitSum) [1..]

firstTo :: Int -> Maybe Int 
firstTo n = find ((==n) . digitSum) [1..]

-- Data.Map

phoneBook :: Map.Map String String -- we provide type and two expected types inside
phoneBook = Map.fromList $ [("betty", "555-340-123"), ("bonnie", "555-342-34"),("wendy", "555-333-33")]

-- find value in map by corresponding key
bettysNumber :: Maybe String
bettysNumber = Map.lookup "betty" phoneBook -- Just "555-340-123"

updatedPhoneBook :: Map.Map String String -- we provide type and two expected types inside
updatedPhoneBook = Map.insert "grace" "341-9021" phoneBook

phoneBookSize :: Int
phoneBookSize = Map.size phoneBook

-- convert phone number strings to list of digits in phoneBook map

string2digits :: String -> [Int]
string2digits = map digitToInt . filter isDigit

phoneBookInt :: Map.Map String [Int]
phoneBookInt = Map.map string2digits phoneBook

---- New data types

-- ShapeS is the new data type
-- CircleS is value constructor, Float Float Float are fields
-- CircleS is a function that takes three Float and returns ShapeS
-- deriving (Show) make this type part of Show type class, to be able to print to console
data ShapeS = CircleS Float Float Float | RectangleS Float Float Float Float deriving (Show)

area :: ShapeS -> Float
-- pattern match against different value constructors
-- we bind its fields to variable names or _
area (CircleS _ _ r) = pi * r ^ 2
area (RectangleS x1 y1 x2 y2) = (abs $ x2 - x1) * (abs $ y2 - y1)

myCircle = CircleS 10 20 10

areaOfCircle = area myCircle -- 314.15927
areaOfRectangle = area $ RectangleS 0 0 10 10 -- 100.0

-- value constructors are functions
listOfCircles = map (CircleS 10 20) [4,5,6] -- [CircleS 10.0 20.0 4.0,CircleS 10.0 20.0 5.0,CircleS 10.0 20.0 6.0]

data Point = Point Float Float deriving (Show)
data Shape = Circle Point Float | Rectangle Point Point deriving (Show) -- Shape better ShapeS implementation

area' :: Shape -> Float
area' (Circle _ r) = pi * r ^ 2
area' (Rectangle (Point x1 y1) (Point x2 y2)) = (abs $ x2 - x1) * (abs $ y2 - y1) -- nested pattern match

-- move shape on x and y axis
nudge :: Shape -> Float -> Float -> Shape
nudge (Circle (Point x y) r) a b = Circle (Point (x+a) (y+b)) r -- new circle with x y coords
nudge (Rectangle (Point x1 y1) (Point x2 y2)) a b = Rectangle (Point (x1+a) (y1+b)) (Point (x2+a) (y1+b))

---- Record syntax

-- company, model, year are field names
data Car = Car { company :: String, model :: String, year :: Int } deriving (Show)

myCar = Car { company="Ford", model="Mustang", year=1967 } -- no proper order of fields
myCarsCompany = company myCar -- automatically created function that look up fields in Car data type

-- pattern matching record syntax
tellCar :: Car -> String
tellCar (Car {company=c, model=m, year=y}) = "This " ++ c ++ " " ++ m ++ " was made in " ++ show y

---- Type parameters

-- Vector has type parameter, its now called type constructor
-- Vector is a parametrized type
-- we want a to be Int, Integer, Double -- therefore type parameter
data Vector a = Vector a a a deriving (Show) 

-- we could include type class to Vector type in data declaration
-- convention not to do this, because then we must repeat it in all functions - also those that don't need it
-- data (Num a) => Vector a = Vector a a a deriving (Show) 

vplus :: (Num a)  => Vector a -> Vector a -> Vector a -- we can now use "a" here
(Vector i j k) `vplus` (Vector l m n) = Vector (i+l) (j+m) (k+n)

dotProd :: (Num a) => Vector a -> Vector a -> a
(Vector i j k) `dotProd` (Vector l m n) = i*l + j*m + k*n 

vmult :: (Num a) => Vector a -> a -> Vector a
(Vector i j k) `vmult` m = Vector (i*m) (j*m) (k*m)

-- when declaring a data type:
-- part before = is the type constructor
-- part after = is value constructor

---- Type synonyms

type PhoneNumber = String
type Name = String
type PhoneBook = [(PhoneNumber, Name)]

inPhoneBook :: Name -> PhoneNumber -> PhoneBook -> Bool -- more readable
inPhoneBook name number book = (name, number) `elem` book

-- parameterized type that represents association list type
-- but any type as key and value
type AssocList k v = [(k, v)]

-- AssocList is a type constructor that takes two types 
-- and produces a concrete type [(Int, String)]
myListOfStuff :: AssocList Int String
myListOfStuff = [(1, "boat"), (2, "car"), (3, "cycle")]

-- NB! we can also partially apply type constructors to produce more types

-- example of type for function getByKey
getByKey :: (Eq k) => k -> AssocList k v -> Maybe v
getByKey _ [] = Nothing
getByKey search ((key, value):restOfList)
  | search == key = Just value
  | search /= key = getByKey search restOfList 

-- Type synonyms (and types generally) can be used only in the type portion of Haskell
-- it includes data and type declarations and after :: in type declarations or type annotations

data LockerState = Taken | Free deriving (Show, Eq)
type Code = String
type LockerMap = Map.Map Int (LockerState, Code)

lockers :: LockerMap
lockers = Map.fromList [
  (101, (Free, "3924")),
  (102, (Taken, "2342")),
  (103, (Free, "3343"))
                       ]

-- Either a b, where Left is error message, and Right is result
-- implementation (| means or):
-- data Either a b = Left a | Right b
lockerLookup :: Int -> LockerMap -> Either String Code
-- map lookup returns Maybe, so Nothing or Just
-- case <expression> of <patterns..>, returns expression
lockerLookup lockerNumber map = case Map.lookup lockerNumber map of
  Nothing -> Left $ "Locker " ++ show lockerNumber ++ " doesn't exist!" 
  Just (state, code) ->  if state /= Taken 
    then Right code
    else Left $ "Locker " ++ show lockerNumber ++ " is taken!"

---- Recursive data types

-- it is either an empty list or a combination of head with some value and rest of list
-- cons means : (list constructor)
-- cons takes value, another list and returns a list
data List' a = Empty' | Cons' a (List' a) deriving (Show, Read, Eq, Ord)
-- data List a = Empty | Cons {listHead :: a, listTail:: List a} deriving (Show, Read, Eq, Ord) -- in record syntax

ourList :: List' Int
ourList = 3 `Cons'` (4 `Cons'` (5 `Cons'` Empty'))
-- equivalent to:
-- 3:(4:(5:[]))

-- functions with only special chars are automatically infix
-- this can be done with value constructors too
-- but infix constructors must begin with :

-- Our own : operator

-- fixity declaration gives functions fixity and defines:
-- how tightly operator binds
-- is it left or right associative 
infixr 5 :-:

data List a = Empty | a :-: (List a) deriving (Show, Read, Eq, Ord)
ourBetterList = 3 :-: 4 :-: 5 :-: Empty

-- Our own ++ operator

infixr 5 ^++
(^++) :: List a -> List a -> List a
Empty ^++ ys = ys
(x :-: xs) ^++ ys = x :-: (xs ^++ ys) -- pattern matching works because it is about matching value constructors

ourListA = 3 :-: 4 :-: 5 :-: Empty
ourListB = 6 :-: 7 :-: Empty
ourListSum = ourListA ^++ ourListB

---- Instances of Type Classes

-- type constructor = value constructor
data TrafficLight = Red | Yellow | Green
-- we are not deriving type classes we will write instances of type classes by hand
-- TrafficLight is a concrete type (without type variable)

-- instance, means make our type (TrafficLight) an instance of the type class Eq
instance Eq TrafficLight where
  -- to fullfil minimal complete definition of Eq we need to overwrite either == or /=
  Red == Red = True
  Yellow == Yellow = True
  Green == Green = True
  _ == _ = False -- == and /= are defined in type class definition of Eq in terms of each other

-- to fullfil minimal comtlete definition of Show we need to implement its show function
instance Show TrafficLight where
  show Red = "Red light"
  show Yellow = "Yellow light"
  show Green = "Green light"

isRedEqualRed = Red == Red
isRedEqualGreen = Red == Green
isRedInside = Red `elem` [Red, Green, Yellow] -- elem works on Eq type class
-- > [Red, Green] -- [Red light, Green light] -- show

---- Custom Type Classes

-- TruthyOrFalsy type class defines one function
-- isTruthy takes one value of a type and considers if it is truthy or falsy
class TruthyOrFalsy a where
  isTruthy :: a -> Bool -- minimal complete definition

-- can add (Num a, Eq a) => constraint, but must define
-- {-# LANGUAGE UndecidableInstances, FlexibleInstances #-} -- https://stackoverflow.com/a/33600361
-- instance (Num a, Eq a) => TruthyOrFalsy a where 
instance TruthyOrFalsy Int where 
  isTruthy 0 = False
  isTruthy _ = True

instance TruthyOrFalsy [a] where
  isTruthy [] = False
  isTruthy _ = True

instance TruthyOrFalsy Bool where
  isTruthy = id -- identity function always returns first argument

instance TruthyOrFalsy (Maybe a) where
  isTruthy (Just _) = True
  isTruthy Nothing = False

instance TruthyOrFalsy (BinaryTree.Tree a) where
  isTruthy BinaryTree.EmptyTree = False
  isTruthy _ = True

instance TruthyOrFalsy TrafficLight where
  isTruthy Red = False
  isTruthy _ = True

ifTruthy :: (TruthyOrFalsy y) => y -> a -> a -> a
ifTruthy value truthyResult falsyResult =
  if isTruthy value
  then truthyResult
  else falsyResult

-- > ifTruthy [] "YEAH!" "NOPE!" -- "NOPE!"
-- > ifTruthy [1,2,3] "YEAH!" "NOPE!" -- "YEAH!"
