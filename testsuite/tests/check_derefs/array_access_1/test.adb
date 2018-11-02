procedure Test is
   type My_Index is range 1 .. 10;
   type My_Array is array (My_Index) of aliased Integer;

   Arr : My_Array;
   X : access all Integer;
begin
   X := Arr (2)'Access;
   if X.all = 42 then
      null;
   end if;
end Ex1;
