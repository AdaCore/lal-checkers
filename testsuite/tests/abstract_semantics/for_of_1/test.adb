procedure Test is
   type Range_Type is range 0 .. 3;
   type Array_Type is array (Range_Type) of Integer;

   My_Array : Array_Type;
   X : Integer := 0;
begin
   My_Array(0) := 12;
   My_Array(1) := 4;
   My_Array(2) := 14;
   My_Array(3) := 7;

   for E of My_Array loop
      X := X + E;
   end loop;
end Ex1;
