import Debug.Trace

doubleMe x = x + x -- function name, parameters, code that makes body of function

doubleUs x y = doubleMe x + doubleMe y
x `doubleUsInfix` y = doubleMe x + doubleMe y -- also as infix function

doubleSmallNumber x = if x > 100
                        then x
                        else x * 2 -- else is mandatory

doubleSmallNumber' x = (if x >100 then x else x * 2) + 1

lostNumbers = [4,8,15,16,23,42]

lostNumbers' = lostNumbers ++ [99] -- append to list

smallCat = 'A' : " SMALL CAT" -- unshift to list

letterB = "Steve Buscemi" !! 6 -- get 6th element
letterB' = (!!) "Steve Buscemi" 6 -- as prefix not infix

big13number = [13,26..] !! 124 -- infinite lists

filterSpecial :: [Char] -> [Char] -- Type declaration -> means returns
filterSpecial string = [ character | character  <- string, character `elem` '_':['0'..'9'] ++ ['a'..'z'] ++ ['A'..'Z'] ]
-- | -- how we want elements to be reflected in resulting list
-- number <- numbers -- we draw elements from this list
-- , -- predicate/condition, one or more

boomBangs numbers = [ if number < 10 then "Boom!" else "Bang!" | number <- numbers, odd number ]

exampleTuple = (3, 1.1, 'a', "hello") -- hold different element types

zigzagTuples = zip [1,2,3] ['a', 'b', 'c'] -- create tuple from lists
zigzagTuples' = zip [1..] ['a', 'b', 'c']

-- example
-- Find right triangle that
-- Sides are integers
-- Length of each side is 0<x<10
-- Sum of all sides is 24

-- generate all possible triples with elements less or equal 10
triples = [ (a,b,c) | c <- [1..10], a <- [1..10], b <- [1..10] ] 
triples' = [ (a,b,c) | c <- [1..10], a <- [1..c], b <- [1..a] ] 

-- generate all possible triples with elements less or equal 10
rightAngles = [ (a,b,c) | c <- [1..10], a <- [1..10], b <- [1..10], a^2 + b^2 == c^2 ] 
rightAngles' = [(a,b,c) | (a,b,c) <- triples', a^2 + b^2 == c^2]

-- apply last condition
rightTriangles = [(a,b,c) | (a,b,c) <- rightAngles', a + b + c == 24]

circumference :: Float -> Float
circumference r = 2 * pi * r

circumference' :: Double -> Double -- More precision
circumference' r = 2 * pi * r

whatIsGreater = "abracadabra" `compare` "zebra" -- LT, GT or EQ? Will return ordering type variable
whatIsGreater' = show whatIsGreater -- Convert to string

-- pattern matching for different function bodies
errorMessage :: Int -> String
errorMessage 404 = "Not Found"
errorMessage 403 = "Forbidden"
errorMessage 501 = "Internal Server Error"
errorMessage i = "Other Error"

-- pattern matching in tuples
third :: (a, b, c) -> c
third (_, _, z) = z

-- pattern inside list comperhension
listOfSumsOfPairs = [a + b | (a, b) <- [(1, 3), (4, 3), (2, 4), (5, 3)]] -- (a, b) is a pattern matched when drawing elements from list

-- pattern for list, our own head implementation
head' :: [a] -> a
head' [] = error "Can't call head on empty list" -- error function dies execution
head' (x:_) = x -- x is first element, _ is rest of the list, paranteses used when binding to several variables (not a tuple)

-- guards
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

-- let..in can be used in other expressions
-- things after let are locally scoped to expression after in
squares = [let square x = x * x; twoMore = 2 in (square 5 + twoMore, square 4 + twoMore)] -- [(27, 18)]

-- let..in used in list comprehansion
-- local variable avaiable in output and predicates not generator
bmiBulk' :: [(Float, Float)] -> [Float]
bmiBulk' listOfPairs = [bmi | (w, h) <- listOfPairs, let bmi = w / h ^ 2]

-- case..of as expression
describeList :: [a] -> String
describeList list = "This list is " ++ case list of [] -> "empty."
                                                    [x] -> "singleton."
                                                    other -> "longer."

-- Returns highest of a list
-- Debug statement from https://stackoverflow.com/a/9849243
maximum' :: (Ord a, Show a) => [a] -> a
maximum' [] = error "Cannot get maximum of empty list"
maximum' [x] = trace ("maximum' [" ++ show x ++ "] == " ++ show x) $ x
maximum' (x:xs) = trace ("max " ++ show x ++ " (maximum' " ++ show xs ++ ")") $ max x (maximum' xs) 
-- Split list into head x and tail (rest) xs
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


-- Returns highest of a list (without using max), my own implementation
maximum'' :: (Ord a, Show a) => [a] -> a
maximum'' [] = error "Cannot get maximum of empty list"
maximum'' [x] = x
maximum'' (x:xs)
  | x > maximum'' xs = x
  | otherwise = maximum'' xs 

-- Example
-- maximum'' [2,5,1]
  -- First two patterns don't match
  -- First guard check: 2 > maximum'' [5, 1] 
    -- Next recursion: maximum'' [5, 1] 
    -- First guard check: 5 > maximum [1]
    -- 5 > 1
    -- True, return 5
  -- Back a level: 2 > 5
  -- First guard check: False
  -- Second guard check:
  -- maximum'' [5] 
  -- Which is 5

-- Replicates value x times in a list
replicate' :: Int -> a -> [a]
replicate' times value 
  | times <= 0 = []
  | otherwise = value : replicate (times-1) value

-- Returns first n elements from provided list
take' :: Int -> [a] -> [a]
take' n _
  | n <= 0 = []
take' _ [] = []
take' n (x:xs) = x : (take' (n-1) xs)

-- Reverse a list
reverse' :: [a] -> [a]
reverse' [] = []
reverse' (x:xs) = (reverse' xs) ++ [x]

-- Zip a list into pairs. 
-- Ex. zip [1,2,3] [7,8] -- [(1,7), (2,8)]
zip' :: [a] -> [b] -> [(a, b)] 
zip' _ [] = []
zip' [] _ = []
zip' (x:xs) (y:ys) = (x, y) : zip' xs ys

-- Checks if element is in the list
elem' :: Eq a => a -> [a] -> Bool
elem' _ [] = False
elem' y (x:xs)
  | x == y = True
  | otherwise = elem' y xs

-- Quicksort
quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) =
  let smallerOrEqual = [n | n <- xs, n <= x ];
      bigger = [n | n <- xs, n > x]
  in quicksort smallerOrEqual ++ [x] ++ quicksort bigger
