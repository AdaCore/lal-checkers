procedure Test is
   type Range_Type is range 0 .. 5;
   type Array_Type is array (Range_Type) of Integer;

   My_Array : Array_Type;
begin
   for E of My_Array loop
      null;
   end loop;
end Ex1;
