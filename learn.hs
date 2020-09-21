import Debug.Trace

doubleMe x = x + x -- function name, parameters, code that makes body of function

---- Infix function

doubleUs x y = doubleMe x + doubleMe y
x `doubleUsInfix` y = doubleMe x + doubleMe y

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

---- Built in functions

-- > succ 5 -- 6 -- next successor
-- > min 5 4 -- 4 
-- > max 5 4 -- 5
-- > div 4 2 -- 2 -- integral division
-- > subtract 5 4 -- -1 -- subtracts 5 from 4

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
-- > cycle [5,4,3] -- [5,4,3,5,4,3,5,4,3 ... (to infinity)]
-- > repeat 5 -- [5,5,5,5 ... (to infinity)]
-- > replicate 3 5 -- [5, 5, 5]

-- > fst (5,4) -- 5 -- only pairs
-- > snd (5,4) -- 4 -- only pairs
-- > zip [5,4,3] ['a', 'b', 'c'] -- [(5, 'a'), (4, 'b'), (3, 'c')] -- zips elements of two lists into pairs

-- > compare 2 4 -- LT -- checks for equality, returns GT LT EQ of Ordering type
-- > show 5 -- "5" -- convert value as string
-- > read "True" :: Bool -- True -- convert string, type must be supplied

-- > flip replicate 3 5 -- [3,3,3,3,3] -- flips next function arguments

---- List comprehensions

filterSpecial :: [Char] -> [Char]
filterSpecial string = [ character | character  <- string, character `elem` '_':['0'..'9'] ++ ['a'..'z'] ++ ['A'..'Z'] ]
-- | -- how we want elements to be reflected in resulting list
-- number <- numbers -- we draw elements from this list
-- , -- predicate/condition, one or more

boomBangs numbers = [ if number < 10 then "Boom!" else "Bang!" | number <- numbers, odd number ]

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

---- Guards

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
describeList list = "This list is " ++ case list of [] -> "empty."
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
-- equivalent to:
flip'' :: (a -> b -> c) -> b -> a -> c
flip'' f x y = f y x
