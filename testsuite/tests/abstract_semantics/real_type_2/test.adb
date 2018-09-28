procedure Test is
   type My_Real is delta 0.01 digits 8 range 0 .. 1;

   X : My_Real := 0.05;
begin
   X := X + 0.01;
end Ex1;
