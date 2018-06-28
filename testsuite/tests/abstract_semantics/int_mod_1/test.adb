procedure Ex1 is
   type My_Int is mod 256;

   X : My_Int := 12;
begin
   X := X + 5;
   X := X + 250;
end Ex1;
