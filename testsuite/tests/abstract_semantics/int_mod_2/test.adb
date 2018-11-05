procedure Test is
   type My_Int is mod 256;

   X : My_Int;
begin
   if X = -1 then
      X := -42;
   end if;
end Ex1;
