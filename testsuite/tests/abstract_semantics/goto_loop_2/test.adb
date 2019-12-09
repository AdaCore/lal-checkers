procedure Test is
   type Arr is array (Integer range <>) of Integer;

   N : Positive;
   A : Arr (1 .. N);

   Tmp : Integer;
begin
   <<sort>>
   for I in 1 .. N - 1 loop
      if A (I) > A (I + 1) then
         Tmp := A (I);
         A (I) := A (I + 1);
         A (I + 1) := Tmp;
         goto sort;
      end if;
   end loop;
end Test;
