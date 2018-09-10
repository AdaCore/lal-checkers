procedure Test is
   type Range_Type is range 0 .. 5;
   type Array_Type is array (Range_Type) of Integer;

   My_Array : Array_Type;
   X : Range_Type := 0;
begin
   for I in My_Array'Range loop
      X := I;
   end loop;
end Ex1;
