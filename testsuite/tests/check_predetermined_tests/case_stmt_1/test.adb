procedure Test is
   type My_Enum is (A, B, C);

   X : My_Enum := A;
   R : Integer;
begin
   case X is
      when A =>
         R := 1;
      when B =>
         R := 2;
      when others =>
         R := 3;
   end case;
end Ex1;
